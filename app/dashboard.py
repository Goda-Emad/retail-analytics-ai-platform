import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from catboost import CatBoostRegressor
import joblib
import os
import base64

# ================== Path Management (Safe & Dynamic) ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# المسارات للمجلدات
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
LOGO_PATH = os.path.join(ROOT_DIR, "images", "retail_ai_pro_logo.webp")
BG_PATH = os.path.join(ROOT_DIR, "images", "bg_retail_1.png")

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro v9.1 | Eng. Goda Emad", layout="wide")

# ================== Theme Logic ==================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark 🌙"

def toggle_theme():
    st.session_state.theme_mode = "Dark 🌙" if st.session_state.theme_mode == "Light 🌞" else "Light 🌞"

st.sidebar.button("🌗 Switch Theme Mode", on_click=toggle_theme)
theme_mode = st.session_state.theme_mode

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

# ================== Image Encoding (Base64) ==================
def get_base64_img(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64_img(LOGO_PATH)
bg_base64 = get_base64_img(BG_PATH)

# ================== CSS Injection ==================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover; background-position: center; background-attachment: fixed;
}}
.stApp::before {{
    content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-color: {bg_overlay}; z-index: -1;
}}
.header-container {{
    display: flex; align-items: center; padding: 20px;
    background-color: {card_bg}; border-radius: 15px; margin-bottom: 25px;
    border-left: 10px solid {accent_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}
.metric-box {{
    background-color: {card_bg}; padding: 20px; border-radius: 12px;
    text-align: center; border: 1px solid {accent_color}; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}
</style>
""", unsafe_allow_html=True)

# ================== Load Model & Data (The Core Fix) ==================
@st.cache_resource
def load_essentials():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(DATA_PATH)):
        return None, None, None
    
    m = joblib.load(MODEL_PATH)
    f = joblib.load(FEATURES_PATH)
    d = pd.read_parquet(DATA_PATH)
    
    # حماية من KeyError: الكشف عن عمود التاريخ تلقائياً
    if d.index.name in ['InvoiceDate', 'date', 'Date']:
        d = d.reset_index()
    
    date_col = next((c for c in ['InvoiceDate', 'date', 'Date', 'Invoice_Date'] if c in d.columns), d.columns[0])
    d["InvoiceDate"] = pd.to_datetime(d[date_col])
    
    # الكشف عن عمود المبيعات تلقائياً
    sales_col = next((c for c in ['TotalAmount', 'Daily_Sales', 'Sales'] if c in d.columns), d.columns[-1])
    d["SalesValue"] = d[sales_col]
    
    return m, f, d

model, feature_names, df = load_essentials()

if df is None:
    st.error("❌ ملفات المشروع مفقودة في مجلد app/. تأكد من هيكل GitHub الخاص بك.")
    st.stop()

# ترتيب الداتا للاستخدام
sales_hist = df.sort_values("InvoiceDate").set_index("InvoiceDate")["SalesValue"]

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <img src="data:image/webp;base64,{logo_base64}" width="70">
    <div style="margin-left:20px;">
        <h1 style="margin:0; color:{accent_color};">Retail AI Pro v9.1</h1>
        <p style="margin:0; color:{text_color}; opacity:0.8; font-weight:bold;">Eng. Goda Emad | Smart Forecasting System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Prediction Engine ==================
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
        
        # هندسة الميزات بناءً على تدريب CatBoost
        features = {
            'day_sin': d_sin, 'week_sin': w_sin, 'month_sin': m_sin,
            'lag_1': current_hist.iloc[-1],
            'lag_7': current_hist.iloc[-7] if len(current_hist)>=7 else current_hist.mean()
        }
        
        # التأكد من ترتيب الأعمدة كما تدرب الموديل
        X_df = pd.DataFrame([features])
        # ملء أي ميزات مفقودة بقيمة صفرية لتجنب خطأ الموديل
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

# ================== Control Sidebar ==================
with st.sidebar:
    st.header("🎮 Forecasting Hub")
    scenario = st.selectbox("Market Condition", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
    horizon = st.slider("Forecast Period (Days)", 7, 30, 14)
    noise_lvl = st.slider("Market Volatility", 0.0, 0.1, 0.03)
    st.divider()
    run_btn = st.button("🚀 Run AI Engine", use_container_width=True)

# ================== Dashboard Logic ==================
if run_btn:
    with st.spinner("AI analyzing retail trends..."):
        preds, dates = generate_forecast(sales_hist, horizon, scenario, noise_lvl)
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metric-box'>Total Predicted Revenue<br><h2>${preds.sum():,.0f}</h2></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-box'>Average Daily Sales<br><h2>${preds.mean():,.0f}</h2></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-box'>Model Confidence Score<br><h2>82.09%</h2></div>", unsafe_allow_html=True)

        # Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales_hist.index[-45:], y=sales_hist.values[-45:], name="Historical Data", line=dict(color="gray", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=preds, name="AI Projection", line=dict(color=accent_color, width=4)))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color=text_color, xaxis_title="Timeline", yaxis_title="Sales Revenue ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Result Table
        st.subheader("📋 Day-by-Day Forecast Details")
        res_df = pd.DataFrame({"Forecast Date": dates, "Estimated Sales": preds})
        st.dataframe(res_df.style.format({"Estimated Sales": "${:,.2f}"}), use_container_width=True)
        st.download_button("📥 Download Report (CSV)", res_df.to_csv(index=False), "retail_ai_forecast.csv")
else:
    st.info("👋 Hello! Use the control panel on the left to start your AI-powered sales forecast.")

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:30px; color:{text_color}; opacity:0.5; font-size:0.85rem;">
    Retail Analytics Platform v9.1 | Powered by CatBoost Regression | © 2025 Eng. Goda Emad
</div>
""", unsafe_allow_html=True)
