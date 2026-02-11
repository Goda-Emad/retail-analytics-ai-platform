# ==================== app.py (Professional Supermarket + Download + Improved Charts) ====================
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
BG_PATH = os.path.join(CURRENT_DIR, "supermarket_bg.jpg")  # ضع صورة الخلفية هنا

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================== Load Model & Data ==================
@st.cache_resource
def load_essentials():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(DATA_PATH)):
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_parquet(DATA_PATH)

    if df.index.name is not None:
        df = df.reset_index()
    
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        st.error(f"❌ لم يتم العثور على عمود تاريخ. الأعمدة: {df.columns.tolist()}")
        st.stop()
    
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

# ================== Dark/Light Mode ==================
mode = st.sidebar.selectbox("اختر وضع الواجهة", ["Dark 🌙", "Light 🌞"])
if mode == "Dark 🌙":
    overlay = "rgba(10, 10, 20, 0.4)"
    text_color = "#ffffff"
    accent_color = "#00D4FF"
    card_bg = "rgba(255, 255, 255, 0.07)"
    border_color = "rgba(255, 255, 255, 0.15)"
else:
    overlay = "rgba(255, 255, 255, 0.4)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255, 255, 255, 0.5)"
    border_color = "rgba(0, 0, 0, 0.1)"

# ================== Load Background ==================
with open(BG_PATH, "rb") as f:
    bg_base64 = base64.b64encode(f.read()).decode()

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
    position: fixed; top:0; left:0; width:100%; height:100%;
    background: {overlay};
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    z-index: -1;
}}
.header-container {{
    display:flex; align-items:center; padding:25px;
    background: {card_bg}; 
    backdrop-filter: blur(10px);
    border-radius:20px; margin-bottom:25px;
    border: 1px solid {border_color};
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}}
.metric-box {{
    background: {card_bg}; 
    backdrop-filter: blur(10px);
    padding:25px; border-radius:15px;
    text-align:center; border: 1px solid {border_color};
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    color: {text_color};
}}
.metric-box h2 {{
    color: {accent_color} !important;
    margin: 10px 0 0 0;
}}
.sidebar-link {{
    display:block; margin-top:5px; color:{accent_color}; font-weight:bold; text-decoration:none;
}}
section[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(15px);
}}
</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <div style="width: 100%; text-align: center;">
        <h1 style="margin:0; color:{accent_color}; font-size: 2.5rem; text-transform: uppercase; letter-spacing: 2px;">Retail AI Pro</h1>
        <p style="margin:5px; color:{text_color}; opacity:0.9; font-weight:400;">Eng. Goda Emad | Intelligent Supermarket Forecasting</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Forecast Functions ==================
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
        if scenario == "متفائل (+15%)": pred *= 1.15
        elif scenario == "متشائم (-15%)": pred *= 0.85
        pred = max(0, pred*(1+np.random.normal(0,noise_val)))
        forecast_values.append(pred)
        current_hist.loc[next_date] = pred
    return np.array(forecast_values), current_hist.index[-horizon:]

# ================== Sidebar ==================
with st.sidebar:
    st.markdown(f"<h2 style='color:{accent_color};'>🛒 Control Panel</h2>", unsafe_allow_html=True)
    scenario = st.selectbox("اختار سيناريو السوق", ["واقعي", "متفائل (+15%)", "متشائم (-15%)"])
    horizon = st.slider("عدد أيام التوقع", 7, 30, 14)
    noise_lvl = st.slider("تقلب السوق", 0.0, 0.1, 0.03)
    start_date = st.date_input("من تاريخ", df.index.min().date())
    end_date = st.date_input("إلى تاريخ", df.index.max().date())
    run_btn = st.button("🚀 تشغيل التوقعات", use_container_width=True)

# ================== Main ==================
if run_btn:
    df_filtered = sales_hist[start_date:end_date]
    
    scenarios_list = ["واقعي", "متفائل (+15%)", "متشائم (-15%)"]
    colors = ["#00D4FF", "#00FF88", "#FF4B2B"]
    
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered.values,
        name="البيانات التاريخية",
        fill='tozeroy',
        fillcolor='rgba(150,150,150,0.1)',
        line=dict(color="rgba(200,200,200,0.5)", width=2),
        hovertemplate="تاريخ: %{x}<br>المبيعات: %{y:.0f}$<extra></extra>"
    ))
    
    # Moving Average 7 days
    df_filtered_ma = df_filtered.rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=df_filtered_ma.index,
        y=df_filtered_ma.values,
        name="متوسط 7 أيام",
        line=dict(color="orange", width=3, dash='dash')
    ))
    
    forecasts_dict = {}
    for sc, color in zip(scenarios_list, colors):
        preds, dates = generate_forecast(df_filtered, horizon, sc, noise_lvl)
        forecasts_dict[sc] = preds
        fig.add_trace(go.Scatter(
            x=dates,
            y=preds,
            name=f"توقع ({sc})",
            mode='lines+markers',
            line=dict(color=color, width=4, shape='spline'),
            marker=dict(size=6),
            hovertemplate="تاريخ: %{x}<br>توقع: %{y:.0f}$<extra></extra>"
        ))
    
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="المبيعات ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # KPI Cards
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    total_forecast = np.mean([forecasts_dict[sc].sum() for sc in scenarios_list])
    avg_forecast = total_forecast / horizon
    c1.markdown(f"<div class='metric-box'>إجمالي المبيعات المتوقعة<br><h2>${total_forecast:,.0f}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'>المعدل اليومي المستهدف<br><h2>${avg_forecast:,.0f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'>دقة نموذج التنبؤ AI<br><h2>82%</h2></div>", unsafe_allow_html=True)

    # ================== Download Button ==================
    def get_csv_download(forecasts_dict, dates):
        df_download = pd.DataFrame({"Date": dates})
        for sc, preds in forecasts_dict.items():
            df_download[sc] = preds
        return df_download

    csv_data = get_csv_download(forecasts_dict, dates).to_csv(index=False).encode()
    st.download_button(
        label="⬇️ تحميل التوقعات CSV",
        data=csv_data,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

else:
    st.markdown(f"""
    <div style="background:{card_bg}; padding:50px; border-radius:20px; text-align:center; border:1px solid {border_color};">
        <h2 style="color:{text_color}; opacity:0.8;">جاهز للتحليل الذكي؟</h2>
        <p style="color:{text_color}; opacity:0.6;">استخدم الشريط الجانبي لتحديد معايير التوقع وتشغيل الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:30px; color:{text_color}; opacity:0.7; font-size:0.9rem;">
    <strong>Developed by Eng. Goda Emad</strong><br>
    <a href='https://www.linkedin.com/in/goda-emad/' class='sidebar-link' style='display:inline;'>LinkedIn</a> | 
    <a href='https://github.com/Goda-Emad' class='sidebar-link' style='display:inline;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
