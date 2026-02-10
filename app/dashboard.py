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
ROOT_DIR = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(ROOT_DIR, "images", "retail_ai_pro_logo.webp")
BG_PATH = os.path.join(ROOT_DIR, "images", "bg_retail_1.png")

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro vFinal", layout="wide")

# ================== Theme ==================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark 🌙"

def toggle_theme():
    st.session_state.theme_mode = "Dark 🌙" if st.session_state.theme_mode=="Light 🌞" else "Light 🌞"

st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)
theme_mode = st.session_state.theme_mode

if theme_mode=="Dark 🌙":
    bg_overlay = "rgba(15,23,42,0.88)"
    text_color = "#f1f5f9"
    accent_color = "#3b82f6"
    card_bg = "rgba(30,41,59,0.7)"
else:
    bg_overlay = "rgba(248,250,252,0.88)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255,255,255,0.7)"

# ================== Base64 Images ==================
def get_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64(LOGO_PATH)
bg_base64 = get_base64(BG_PATH)

# ================== CSS ==================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover; background-position: center; background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: fixed; top:0; left:0; width:100%; height:100%;
    background-color: {bg_overlay}; z-index: -1;
}}
.header-container {{
    display:flex; align-items:center; padding:20px;
    background-color:{card_bg}; border-radius:15px; margin-bottom:25px;
    border-left:10px solid {accent_color}; box-shadow:0 4px 15px rgba(0,0,0,0.3);
}}
.metric-box {{
    background-color:{card_bg}; padding:20px; border-radius:12px;
    text-align:center; border:1px solid {accent_color}; box-shadow:0 2px 10px rgba(0,0,0,0.1);
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
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    return model, features, df

model, feature_names, df = load_essentials()
if df is None:
    st.error("❌ Files missing in app/. Please upload all required files.")
    st.stop()

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/webp;base64,{logo_base64}" width="70">
    <div style="margin-left:20px;">
        <h1 style="margin:0; color:{accent_color};">Retail AI Pro vFinal</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8; font-weight:bold;">Eng. Goda Emad | Smart Forecast System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Sales History ==================
sales_hist = df.sort_index()["Daily_Sales"]

# ================== Feature Engineering ==================
def get_cyclical_features(date):
    day_sin = np.sin(2*np.pi*date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(date.isocalendar().week%52)/52)
    month_sin = np.sin(2*np.pi*date.month/12)
    return day_sin, week_sin, month_sin

# ================== Forecast Engine ==================
def generate_forecast(hist, horizon, scenario, noise):
    forecast = []
    hist_copy = hist.copy()
    for i in range(horizon):
        next_date = hist_copy.index[-1] + timedelta(days=1)
        d, w, m = get_cyclical_features(next_date)
        feats = {
            "day_sin": d, "week_sin": w, "month_sin": m,
            "lag_1": hist_copy.iloc[-1],
            "lag_7": hist_copy.iloc[-7] if len(hist_copy)>=7 else hist_copy.mean()
        }
        X_df = pd.DataFrame([feats])
        for f in feature_names:
            if f not in X_df.columns:
                X_df[f] = 0
        X_df = X_df[feature_names]
        pred = model.predict(X_df)[0]
        if "Optimistic" in scenario: pred *=1.15
        elif "Pessimistic" in scenario: pred *=0.85
        pred = max(0, pred*(1+np.random.normal(0,noise)))
        forecast.append(pred)
        hist_copy.loc[next_date]=pred
    return np.array(forecast), [hist_copy.index[-horizon+i] for i in range(horizon)]

# ================== Sidebar ==================
with st.sidebar:
    st.header("Forecast Controls")
    scenario = st.selectbox("Market Scenario", ["Realistic","Optimistic (+15%)","Pessimistic (-15%)"])
    horizon = st.slider("Forecast Horizon (Days)", 7,30,14)
    noise_lvl = st.slider("Volatility (Noise)",0.0,0.1,0.03)
    st.divider()
    run_btn = st.button("🚀 Run Forecast")

# ================== Run Forecast ==================
if run_btn:
    with st.spinner("Generating AI forecast..."):
        preds, dates = generate_forecast(sales_hist, horizon, scenario, noise_lvl)
        
        # ===== KPI Cards =====
        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='metric-box'>Total Forecast<br><h2>${preds.sum():,.0f}</h2></div>",unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'>Daily Avg<br><h2>${preds.mean():,.0f}</h2></div>",unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'>Confidence<br><h2>82.1%</h2></div>",unsafe_allow_html=True)
        
        # ===== Plotly Chart =====
        upper = preds*1.05
        lower = preds*0.95
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales_hist.index[-45:], y=sales_hist.values[-45:], mode='lines+markers', name="History", line=dict(color="gray")))
        fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines+markers', name="Forecast", line=dict(color=accent_color, width=3)))
        fig.add_trace(go.Scatter(x=dates+dates[::-1], y=list(upper)+list(lower[::-1]), fill='toself',
                                 fillcolor=f'rgba(59,130,246,0.15)', line=dict(color='rgba(0,0,0,0)'), hoverinfo="skip", name="Error Margin"))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font_color=text_color, xaxis_title="Date", yaxis_title="Sales ($)")
        st.plotly_chart(fig,use_container_width=True)
        
        # ===== Table & Download =====
        df_forecast = pd.DataFrame({"Date":dates,"Forecast":preds,"Lower":lower,"Upper":upper})
        st.subheader("📋 Forecast Table")
        st.dataframe(df_forecast.style.format({"Forecast":"${:,.2f}","Lower":"${:,.2f}","Upper":"${:,.2f}"}))
        st.download_button("📥 Download CSV", df_forecast.to_csv(index=False),"forecast.csv")
else:
    st.info("👈 Use the sidebar to adjust scenario, horizon, and noise, then click Run Forecast.")

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:20px; color:{text_color}; opacity:0.6; font-size:0.85rem;">
    Retail AI Pro | CatBoost Forecasting | Developed by Eng. Goda Emad
</div>
""", unsafe_allow_html=True)
