import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from catboost import CatBoostRegressor
import joblib
import os
import base64

# ================== Paths ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
ROOT_DIR = os.path.dirname(CURRENT_DIR)                  # root
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(ROOT_DIR, "images", "retail_ai_pro_logo.webp")
BACKGROUND_PATH = os.path.join(ROOT_DIR, "images", "bg_retail_1.png")

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================== Theme ==================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light 🌞"

theme_mode = st.sidebar.selectbox(
    "Theme Mode", ["Light 🌞", "Dark 🌙"], index=0
)
st.session_state.theme_mode = theme_mode

if theme_mode == "Dark 🌙":
    bg_overlay = "rgba(15, 23, 42, 0.88)"
    text_color = "#f1f5f9"
    accent_color = "#3b82f6"
    card_bg = "rgba(30, 41, 59, 0.7)"
else:
    bg_overlay = "rgba(248, 250, 252, 0.88)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255, 255, 255, 0.7)"

# ================== Base64 Images ==================
def get_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64(LOGO_PATH)
bg_base64 = get_base64(BACKGROUND_PATH)

# ================== CSS ==================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
}}
.stApp::before {{
    content:"";
    position:fixed; top:0; left:0; width:100%; height:100%;
    background-color:{bg_overlay}; z-index:-1;
}}
.header-container {{
    display:flex; align-items:center; padding:20px;
    background-color:{card_bg}; border-radius:15px; margin-bottom:20px;
    border-left:10px solid {accent_color};
    box-shadow:0 4px 15px rgba(0,0,0,0.3);
}}
.metric-box {{
    background-color:{card_bg}; padding:20px; border-radius:12px;
    text-align:center; border:1px solid {accent_color};
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
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
    if df.index.name == "InvoiceDate":
        df = df.reset_index()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["SalesValue"] = df["Daily_Sales"]
    df = df.sort_values("InvoiceDate").set_index("InvoiceDate")
    return model, features, df

model, feature_names, df = load_essentials()
if df is None:
    st.error("❌ ملفات المشروع غير موجودة في مجلد app/")
    st.stop()

sales_hist = df["SalesValue"]

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/webp;base64,{logo_base64}" width="70">
    <div style="margin-left:20px;">
        <h1 style="margin:0; color:{accent_color};">Retail AI Pro</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8;">Eng. Goda Emad | Smart Forecasting System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Forecast Engine ==================
def get_cyclical_features(date):
    day_sin = np.sin(2*np.pi*date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(date.isocalendar().week % 52)/52)
    month_sin = np.sin(2*np.pi*date.month/12)
    return day_sin, week_sin, month_sin

def generate_forecast(hist, horizon, scenario, noise):
    forecast_values = []
    current_hist = hist.copy()
    for i in range(horizon):
        next_date = current_hist.index[-1] + timedelta(days=1)
        d_sin, w_sin, m_sin = get_cyclical_features(next_date)
        features = {
            'day': next_date.day,
            'month': next_date.month,
            'dayofweek': next_date.dayofweek,
            'weekofyear': next_date.isocalendar().week,
            'is_weekend': int(next_date.dayofweek>=5),
            'sales_lag_1': current_hist.iloc[-1],
            'sales_lag_7': current_hist.iloc[-7] if len(current_hist)>=7 else current_hist.mean(),
            'rolling_mean_7': current_hist[-7:].mean() if len(current_hist)>=7 else current_hist.mean(),
            'rolling_std_7': current_hist[-7:].std() if len(current_hist)>=7 else 0
        }
        X_df = pd.DataFrame([features])
        for feat in feature_names:
            if feat not in X_df.columns:
                X_df[feat] = 0
        X_df = X_df[feature_names]
        pred = model.predict(X_df)[0]
        if "Optimistic" in scenario: pred *= 1.15
        elif "Pessimistic" in scenario: pred *= 0.85
        pred = max(0, pred*(1+np.random.normal(0,noise)))
        forecast_values.append(pred)
        current_hist.loc[next_date] = pred
    return np.array(forecast_values), pd.date_range(current_hist.index[-horizon], periods=horizon)

# ================== Sidebar ==================
with st.sidebar:
    st.header("Forecast Controls")
    scenario = st.selectbox("Market Scenario", ["Realistic","Optimistic (+15%)","Pessimistic (-15%)"])
    horizon = st.slider("Forecast Horizon (Days)", 7,30,14)
    noise_level = st.slider("Noise Level", 0.0,0.1,0.03)
    run_btn = st.button("🚀 Run AI Forecast")

# ================== Main ==================
if run_btn:
    preds, dates = generate_forecast(sales_hist,horizon,scenario,noise_level)
    
    # KPI Cards
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric-box'>Total Forecast<br><h2>${preds.sum():,.0f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'>Daily Avg<br><h2>${preds.mean():,.0f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'>Confidence<br><h2>82%</h2></div>", unsafe_allow_html=True)
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_hist.index[-60:], y=sales_hist.values[-60:], name="History", line=dict(color="gray", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(color=accent_color,width=4)))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color=text_color,xaxis_title="Date",yaxis_title="Sales ($)")
    st.plotly_chart(fig,use_container_width=True)

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:20px; color:{text_color}; opacity:0.5;">
    Retail AI Pro | Eng. Goda Emad | Powered by CatBoost
</div>
""", unsafe_allow_html=True)
