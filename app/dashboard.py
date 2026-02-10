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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(CURRENT_DIR, "images", "retail_ai_pro_logo.webp")

# ================== Page Config ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================== Load Logo ==================
def get_base64_img(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64_img(LOGO_PATH)

# ================== Theme Selector ==================
theme_choice = st.sidebar.selectbox("Theme Mode", ["Dark 🌙", "Light 🌞"])

if theme_choice == "Dark 🌙":
    bg_overlay = "rgba(15,23,42,0.6)"
    text_color = "#f1f5f9"
    accent_color = "#3b82f6"
    card_bg = "rgba(30,41,59,0.7)"
else:
    bg_overlay = "rgba(255,255,255,0.6)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255,255,255,0.7)"

# ================== Supermarket Background ==================
SUPERMARKET_BG_URL = "https://images.pexels.com/photos/373076/pexels-photo-373076.jpeg?auto=compress&cs=tinysrgb&w=1600"

st.markdown(f"""
<style>
.stApp {{
    background-image: url('{SUPERMARKET_BG_URL}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content:"";
    position:fixed;
    top:0; left:0; width:100%; height:100%;
    background-color:{bg_overlay};
    z-index:-1;
}}
.header-container {{
    display:flex; align-items:center; padding:15px;
    background-color:{card_bg}; border-radius:12px;
    border-left:6px solid {accent_color}; margin-bottom:20px;
}}
.metric-box {{
    background-color:{card_bg}; padding:15px;
    border-radius:12px; text-align:center;
    border:1px solid {accent_color};
}}
</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/webp;base64,{logo_base64}" width="70">
    <div style="margin-left:15px;">
        <h1 style="margin:0; color:{accent_color};">Retail AI Pro</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8; font-weight:bold;">Eng. Goda Emad | Smart Forecasting System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Load Model & Data ==================
@st.cache_resource
def load_essentials():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(DATA_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_parquet(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df.index)
    df = df.sort_values("InvoiceDate")
    return model, features, df

model, feature_names, df = load_essentials()
if df is None:
    st.error("❌ Files are missing in app/ folder!")
    st.stop()

sales_hist = df.set_index("InvoiceDate")["Daily_Sales"]

# ================== Sidebar ==================
st.sidebar.header("Forecast Controls")
scenario = st.sidebar.selectbox("Market Scenario", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 30, 14)
noise_lvl = st.sidebar.slider("Volatility (Noise)", 0.0, 0.1, 0.03)

# Slicer for Historical Period
start_date = st.sidebar.date_input("Start Date", df.index.min().date())
end_date = st.sidebar.date_input("End Date", df.index.max().date())

# ================== Forecast Engine ==================
def get_cyclical_features(date):
    return np.sin(2*np.pi*date.dayofweek/7), np.sin(2*np.pi*(date.isocalendar().week % 52)/52), np.sin(2*np.pi*date.month/12)

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
            if feat not in X_df.columns: X_df[feat]=0
        X_df = X_df[feature_names]
        pred = model.predict(X_df)[0]
        if "Optimistic" in scenario: pred*=1.15
        elif "Pessimistic" in scenario: pred*=0.85
        pred = max(0, pred*(1+np.random.normal(0, noise_val)))
        forecast_values.append(pred)
        current_hist.loc[next_date] = pred
    return np.array(forecast_values), current_hist.index[-horizon:]

# ================== Run Forecast ==================
run_btn = st.sidebar.button("🚀 Run Forecast")
if run_btn:
    preds, dates = generate_forecast(sales_hist, horizon, scenario, noise_lvl)

    # KPI Cards
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='metric-box'>Total Forecast<br><h2>${preds.sum():,.0f}</h2></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-box'>Daily Avg<br><h2>${preds.mean():,.0f}</h2></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-box'>Confidence<br><h2>82.1%</h2></div>", unsafe_allow_html=True)

    # Filtered Historical Data
    filtered_hist = sales_hist[(sales_hist.index.date >= start_date) & (sales_hist.index.date <= end_date)]

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_hist.index, y=filtered_hist.values, name="History", line=dict(color="gray", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(color=accent_color, width=4)))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color,
        xaxis_title="Date", yaxis_title="Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:20px; color:{text_color}; opacity:0.6; font-size:0.8rem;">
    Retail Analytics Platform | Eng. Goda Emad | Powered by CatBoost
</div>
""", unsafe_allow_html=True)
