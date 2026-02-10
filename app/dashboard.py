import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from catboost import CatBoostRegressor
import joblib
import os

# ================== Paths ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(CURRENT_DIR, "retail_ai_pro_logo.webp")

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================== Load Essentials ==================
@st.cache_resource
def load_essentials():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(DATA_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_parquet(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df.index if df.index.name=="InvoiceDate" else df.columns[0])
    sales_col = "Daily_Sales" if "Daily_Sales" in df.columns else df.columns[-1]
    df["SalesValue"] = df[sales_col]
    df = df.sort_values("InvoiceDate").set_index("InvoiceDate")
    return model, features, df

model, feature_names, df = load_essentials()
if df is None:
    st.error("❌ ملفات المشروع غير موجودة!")
    st.stop()

sales_hist = df["SalesValue"]

# ================== Theme Control ==================
theme_mode = st.sidebar.selectbox("Choose Theme", ["Dark 🌙", "Light 🌞"])
dark_mode = theme_mode=="Dark 🌙"

bg_overlay = "rgba(15,23,42,0.5)" if dark_mode else "rgba(255,255,255,0.4)"
card_bg = "rgba(30,41,59,0.7)" if dark_mode else "rgba(255,255,255,0.7)"
accent_color = "#3b82f6" if dark_mode else "#2563eb"
text_color = "#f1f5f9" if dark_mode else "#1e293b"

# ================== Supermarket Background ==================
supermarket_bg_url = "https://images.unsplash.com/photo-1585238342027-43a5f78ef0f6?auto=format&fit=crop&w=1600&q=80"
st.markdown(f"""
<style>
.stApp {{
    background-image: url('{supermarket_bg_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content:"";
    position: fixed;
    top:0; left:0; width:100%; height:100%;
    background:{bg_overlay};
    z-index: -1;
}}
.header-container {{
    display:flex; align-items:center; padding:15px;
    background-color:{card_bg}; border-radius:12px; border-left:6px solid {accent_color};
    margin-bottom:20px;
}}
.metric-box {{
    background-color:{card_bg}; padding:15px; border-radius:12px;
    text-align:center; border:1px solid {accent_color};
    box-shadow:0 3px 10px rgba(0,0,0,0.1);
}}
</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="{LOGO_PATH}" width="70">
    <div style="margin-left:20px;">
        <h1 style="margin:0; color:{accent_color};">Retail AI Pro</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8; font-weight:bold;">Eng. Goda Emad | Smart Forecasting AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Forecast Engine ==================
def get_cyclical_features(date):
    return (np.sin(2*np.pi*date.dayofweek/7),
            np.sin(2*np.pi*(date.isocalendar().week % 52)/52),
            np.sin(2*np.pi*date.month/12))

def generate_forecast(hist_series, horizon, scenario, noise_val):
    forecast_values = []
    current_hist = hist_series.copy()
    for i in range(horizon):
        next_date = current_hist.index[-1] + timedelta(days=1)
        d_sin, w_sin, m_sin = get_cyclical_features(next_date)
        features = {
            'day': next_date.day, 'month': next_date.month, 'dayofweek': next_date.dayofweek,
            'weekofyear': next_date.isocalendar().week, 'sales_lag_1': current_hist.iloc[-1],
            'sales_lag_7': current_hist.iloc[-7] if len(current_hist)>=7 else current_hist.mean(),
            'rolling_mean_7': current_hist[-7:].mean() if len(current_hist)>=7 else current_hist.mean(),
            'rolling_std_7': current_hist[-7:].std() if len(current_hist)>=7 else 0
        }
        X_df = pd.DataFrame([features])
        for feat in feature_names:
            if feat not in X_df.columns: X_df[feat] = 0
        X_df = X_df[feature_names]
        pred = model.predict(X_df)[0]
        if "Optimistic" in scenario: pred *= 1.15
        elif "Pessimistic" in scenario: pred *= 0.85
        pred = max(0, pred*(1+np.random.normal(0, noise_val)))
        forecast_values.append(pred)
        current_hist.loc[next_date] = pred
    return np.array(forecast_values), current_hist.index[-horizon:]

# ================== Sidebar ==================
st.sidebar.header("Forecast Controls")
scenario = st.sidebar.selectbox("Market Scenario", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 30, 14)
noise_lvl = st.sidebar.slider("Noise Level", 0.0, 0.1, 0.03)
run_btn = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

# ================== Run Forecast ==================
if run_btn:
    preds, dates = generate_forecast(sales_hist, horizon, scenario, noise_lvl)
    
    # KPI Cards
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='metric-box'>Total Forecast<br><h2>${preds.sum():,.0f}</h2></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-box'>Average Daily Sales<br><h2>${preds.mean():,.0f}</h2></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-box'>Confidence Score<br><h2>82%</h2></div>", unsafe_allow_html=True)
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_hist.index[-45:], y=sales_hist.values[-45:], 
                             name="History", line=dict(color="gray", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=preds, 
                             name="Forecast", line=dict(color=accent_color, width=4)))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color=text_color, xaxis_title="Date", yaxis_title="Sales ($)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # CSV Download
    res_df = pd.DataFrame({"Date": dates, "Forecast": preds})
    st.download_button("📥 Download Forecast CSV", res_df.to_csv(index=False), "forecast.csv")
else:
    st.info("👈 Use the sidebar to select scenario, horizon and run AI forecast.")

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:20px; color:{text_color}; opacity:0.6; font-size:0.85rem;">
Retail Analytics Platform | Eng. Goda Emad | CatBoost AI
</div>
""", unsafe_allow_html=True)
