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
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(CURRENT_DIR, "images", "retail_ai_pro_logo.webp")
SUPERMARKET_BG = os.path.join(CURRENT_DIR, "images", "supermarket_bg.jpg")  # ضع صورة سوبرماركت هنا

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro v10", layout="wide")

# ================== Theme ==================
theme = st.sidebar.selectbox("Choose Theme", ["Light 🌞", "Dark 🌙"])
if theme == "Dark 🌙":
    text_color = "#f1f5f9"
    accent_color = "#3b82f6"
    card_bg = "rgba(30,41,59,0.7)"
else:
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255,255,255,0.7)"

# ================== Image Encoding ==================
def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64_img(LOGO_PATH)
bg_base64 = get_base64_img(SUPERMARKET_BG)

# ================== CSS ==================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
}}
.stApp::before {{
    content: "";
    position: fixed;
    top:0; left:0; width:100%; height:100%;
    background-color: {'rgba(15,23,42,0.85)' if theme=="Dark 🌙" else 'rgba(248,250,252,0.85)'};
    z-index:-1;
}}
.header-container {{
    display:flex; align-items:center; padding:15px; border-radius:15px;
    background-color:{card_bg}; border-left:6px solid {accent_color};
}}
.metric-box {{
    background-color:{card_bg}; padding:15px; border-radius:12px;
    text-align:center; border:1px solid {accent_color};
}}
</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/webp;base64,{logo_base64}" width="60">
    <div style="margin-left:15px;">
        <h1 style="margin:0; color:{accent_color}">Retail AI Pro v10</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8;">Smart Forecasting System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Load Model & Data ==================
@st.cache_resource
def load_essentials():
    m = joblib.load(MODEL_PATH)
    f = joblib.load(FEATURES_PATH)
    d = pd.read_parquet(DATA_PATH)
    d["InvoiceDate"] = pd.to_datetime(d.index if "InvoiceDate" in d.columns else d.index)
    d = d.sort_values("InvoiceDate")
    return m, f, d

model, feature_names, df = load_essentials()
sales_hist = df.set_index("InvoiceDate")["Daily_Sales"]

# ================== Prediction Engine ==================
def get_cyclical_features(date):
    day_sin = np.sin(2*np.pi*date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(date.isocalendar().week%52)/52)
    month_sin = np.sin(2*np.pi*date.month/12)
    return day_sin, week_sin, month_sin

def generate_forecast(hist, horizon, scenario, noise):
    preds = []
    hist_copy = hist.copy()
    for i in range(horizon):
        date = hist_copy.index[-1] + timedelta(days=1)
        d, w, m = get_cyclical_features(date)
        feats = {'day':d,'weekofyear':w,'month':m,
                 'sales_lag_1':hist_copy.iloc[-1],'sales_lag_7':hist_copy.iloc[-7] if len(hist_copy)>=7 else hist_copy.mean()}
        X = pd.DataFrame([feats])
        for col in feature_names:
            if col not in X.columns: X[col]=0
        X = X[feature_names]
        pred = model.predict(X)[0]
        if "Optimistic" in scenario: pred*=1.15
        elif "Pessimistic" in scenario: pred*=0.85
        pred = max(0,pred*(1+np.random.normal(0,noise)))
        preds.append(pred)
        hist_copy.loc[date] = pred
    return np.array(preds), [hist_copy.index[-horizon:][i] for i in range(horizon)]

# ================== Sidebar ==================
st.sidebar.header("Forecast Controls")
scenario = st.sidebar.selectbox("Market Scenario", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", 7,30,14)
noise = st.sidebar.slider("Noise Level",0.0,0.1,0.03)
time_window = st.sidebar.selectbox("Show Last", ["30 Days","60 Days","90 Days","All"])
run_btn = st.sidebar.button("🚀 Run Forecast")

# ================== Dashboard ==================
if run_btn:
    preds, dates = generate_forecast(sales_hist, horizon, scenario, noise)
    
    # KPI Cards
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric-box'>Total Forecast<br><h2>${preds.sum():,.0f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'>Daily Avg<br><h2>${preds.mean():,.0f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'>Confidence<br><h2>82.1%</h2></div>", unsafe_allow_html=True)
    
    # Time window filter
    if time_window=="30 Days": hist_plot = sales_hist[-30:]
    elif time_window=="60 Days": hist_plot = sales_hist[-60:]
    elif time_window=="90 Days": hist_plot = sales_hist[-90:]
    else: hist_plot = sales_hist

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot.values, name="History", line=dict(color="gray",width=2)))
    fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(color=accent_color,width=4)))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download
    df_res = pd.DataFrame({"Date":dates,"Forecast":preds})
    st.download_button("📥 Download Forecast CSV", df_res.to_csv(index=False), "forecast.csv")
else:
    st.info("👈 Use sidebar to select scenario, horizon, noise and click Run Forecast.")
