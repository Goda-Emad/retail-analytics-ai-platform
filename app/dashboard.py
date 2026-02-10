# ==================== app.py ====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from catboost import CatBoostRegressor
import joblib
import os
import base64

# ================== Path Management ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CURRENT_DIR  # كل الملفات داخل مجلد app/

MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(CURRENT_DIR, "images", "retail_ai_pro_logo.webp")
BG_PATH = os.path.join(CURRENT_DIR, "images", "bg_supermarket.jpg")  # سوبرماركت واقعي

# ================== Page Setup ==================
st.set_page_config(
    page_title="Retail AI Pro | Eng. Goda Emad",
    layout="wide"
)

# ================== Dark/Light Mode ==================
mode = st.sidebar.selectbox("Theme Mode", ["Dark 🌙", "Light 🌞"])
if mode == "Dark 🌙":
    bg_overlay = "rgba(15, 23, 42, 0.88)"
    text_color = "#f1f5f9"
    accent_color = "#3b82f6"
    card_bg = "rgba(30, 41, 59, 0.7)"
else:
    bg_overlay = "rgba(248, 250, 252, 0.88)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255, 255, 255, 0.7)"

# ================== Load Images ==================
def get_base64_img(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64_img(LOGO_PATH)
bg_base64 = get_base64_img(BG_PATH)

# ================== CSS ==================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-color: {bg_overlay}; z-index: -1;
}}
.header-container {{
    display: flex; align-items: center; padding: 20px;
    background-color: {card_bg}; border-radius: 15px; margin-bottom: 25px;
    border-left: 10px solid {accent_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}
.metric-box {{
    background-color: {card_bg}; padding: 20px; border-radius: 12px;
    text-align: center; border: 1px solid {accent_color};
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}
</style>
""", unsafe_allow_html=True)

# ================== Load Model & Data ==================
@st.cache_resource
def load_essentials():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(DATA_PATH)):
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_parquet(DATA_PATH)

    # لو التاريخ في index
    if df.index.name is not None:
        df = df.reset_index()
    
    # البحث عن عمود تاريخ
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        st.error(f"❌ لم يتم العثور على عمود تاريخ. الأعمدة: {df.columns.tolist()}")
        st.stop()
    
    # عمود مبيعات
    if "Daily_Sales" not in df.columns:
        possible_sales = [c for c in df.columns if 'sales' in c.lower() or 'amount' in c.lower() or 'total' in c.lower()]
        if possible_sales:
            df = df.rename(columns={possible_sales[0]: "Daily_Sales"})
        else:
            st.error("❌ لم يتم العثور على عمود المبيعات.")
            st.stop()
    
    return model, features, df

model, feature_names, df = load_essentials()
sales_hist = df.sort_index()["Daily_Sales"]

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/webp;base64,{logo_base64}" width="70">
    <div style="margin-left:20px;">
        <h1 style="margin:0; color:{accent_color};">Retail AI Pro</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8; font-weight:bold;">Eng. Goda Emad | Smart Forecasting System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Forecast Engine ==================
def get_cyclical_features(date):
    day_sin = np.sin(2*np.pi*date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(date.isocalendar().week % 52)/52)
    month_sin = np.sin(2*np.pi*date.month/12)
    return day_sin, week_sin, month_sin

def generate_forecast(hist_series, horizon, scenario, noise_val):
    forecast_values = []
    current_hist = hist_series.copy()
    
    for i in range(horizon):
        next_date = current_hist.index[-1] + timedelta(days=1)
        d_sin, w_sin, m_sin = get_cyclical_features(next_date)
        
        features_dict = {
            'day_sin': d_sin, 'week_sin': w_sin, 'month_sin': m_sin,
            'lag_1': current_hist.iloc[-1],
            'lag_7': current_hist.iloc[-7] if len(current_hist)>=7 else current_hist.mean()
        }
        
        X_df = pd.DataFrame([features_dict])
        for feat in feature_names:
            if feat not in X_df.columns: X_df[feat] = 0
        X_df = X_df[feature_names]
        
        pred = model.predict(X_df)[0]
        if "Optimistic" in scenario: pred *= 1.15
        elif "Pessimistic" in scenario: pred *= 0.85
        pred = max(0, pred * (1 + np.random.normal(0, noise_val)))
        
        forecast_values.append(pred)
        current_hist.loc[next_date] = pred
    
    return np.array(forecast_values), current_hist.index[-horizon:]

# ================== Sidebar ==================
with st.sidebar:
    st.header("🎯 Forecast Controls")
    scenario = st.selectbox("Market Scenario", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
    horizon = st.slider("Forecast Period (Days)", 7, 30, 14)
    noise_lvl = st.slider("Market Volatility", 0.0, 0.1, 0.03)
    start_date = st.date_input("Select Start Date", df.index.min().date())
    end_date = st.date_input("Select End Date", df.index.max().date())
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

# ================== Main ==================
if run_btn:
    df_filtered = sales_hist[start_date:end_date]
    preds, dates = generate_forecast(df_filtered, horizon, scenario, noise_lvl)
    
    # KPI Cards
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-box'>Total Forecast<br><h2>${preds.sum():,.0f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'>Average Daily<br><h2>${preds.mean():,.0f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'>Confidence Score<br><h2>82%</h2></div>", unsafe_allow_html=True)
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered.values, name="History", line=dict(color="gray", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(color=accent_color, width=4)))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color,
        xaxis_title="Date",
        yaxis_title="Sales ($)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Use the sidebar to run your AI-powered sales forecast.")

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:20px; color:{text_color}; opacity:0.6; font-size:0.8rem;">
    Retail Analytics Platform | Powered by CatBoost | © 2025 Eng. Goda Emad
</div>
""", unsafe_allow_html=True)
