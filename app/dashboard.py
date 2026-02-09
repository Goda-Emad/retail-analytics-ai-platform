# ==============================
# Retail Sales Forecasting AI
# Developed by Eng. Goda Emad
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from pathlib import Path
import plotly.graph_objects as go

# ---------- 1️⃣ إعداد الصفحة ----------
st.set_page_config(page_title="Retail AI Forecast", layout="wide")

# ---------- 2️⃣ التنسيق الجمالي (CSS) ----------
def apply_css():
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg,#eef2f7,#ffffff);}
    .header-card {
        background:white;padding:35px;border-radius:20px;
        box-shadow:0 10px 30px rgba(0,0,0,0.08);
        text-align:center;margin-bottom:30px;
    }
    .name-title{font-size:44px;font-weight:900;color:#0f172a;}
    .project-title{font-size:26px;font-weight:700;color:#2563eb;}
    .metric-card{
        background:white;padding:25px;border-radius:18px;
        text-align:center;box-shadow:0 8px 20px rgba(0,0,0,0.08);
    }
    .metric-value{font-size:34px;font-weight:700;color:#2563eb;}
    .metric-label{color:#64748b;}
    .stButton>button{
        background:#2563eb;color:white;
        border-radius:10px;height:55px;font-size:18px;width:100%;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- 3️⃣ واجهة العنوان ----------
def render_header():
    st.markdown("""
    <div class='header-card'>
        <div class='name-title'>Eng. Goda Emad</div>
        <div class='project-title'>Smart Retail Sales Forecasting AI</div>
        <p>Interactive ML Dashboard for Predicting Future Retail Sales</p>
        <a href='https://www.linkedin.com/in/goda-emad/' target='_blank'>LinkedIn</a> |
        <a href='https://github.com/Goda-Emad' target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ---------- 4️⃣ إعداد المسارات بناءً على صورك ----------
BASE_DIR = Path(__file__).resolve().parent.parent

# الأسماء مطابقة للصور بالضبط
DATA_PATH = BASE_DIR / "data" / "daily_sales_ready.parquet"
MODEL_PATH = BASE_DIR / "model" / "catboost_sales_model_v2.pkl"
FEAT_PATH = BASE_DIR / "model" / "feature_names.pkl"

# ---------- 5️⃣ تحميل البيانات والموديل ----------
@st.cache_data
def load_essentials():
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error(f"❌ ملفات مفقودة! تأكد من وجود الموديل v2 والبيانات في GitHub.")
        st.stop()
    
    # تحميل البيانات
    df = pd.read_parquet(DATA_PATH)
    # معالجة أسماء الأعمدة لتناسب الكود
    if 'InvoiceDate' in df.columns:
        df.rename(columns={'InvoiceDate': 'date'}, inplace=True)
    if 'total_amount' in df.columns:
        df.rename(columns={'total_amount': 'sales'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    
    # تحميل الموديل وقائمة الميزات
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_PATH)
    
    return df, model, feature_names

df, model, feature_names = load_essentials()

# ---------- 6️⃣ منطق التوقع الديناميكي (v2) ----------
def generate_forecast(model, df_hist, features_list, start_lag1, days=30):
    future_preds = []
    last_date = df_hist['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # نحتاج لآخر 30 مبيعة لحساب المتوسط المتحرك
    history = list(df_hist['sales'].tail(30))
    history[-1] = start_lag1 # تحديث القيمة بناءً على إدخال العميل
    
    for i in range(days):
        d = future_dates[i]
        # بناء الميزات بناءً على ما تدرب عليه الموديل v2
        feat_dict = {
            'day': d.day,
            'month': d.month,
            'dayofweek': d.dayofweek(),
            'is_weekend': 1 if d.dayofweek() in [4, 5] else 0,
            'rolling_mean_7': np.mean(history[-7:]),
            'lag_1': history[-1],
            'lag_7': history[-7] if len(history) >= 7 else history[-1],
            'day_of_year': d.dayofyear,
            'week_of_year': d.isocalendar()[1]
        }
        
        # ترتيب الميزات كما يتوقع الموديل
        input_data = [feat_dict.get(f, 0) for f in features_list]
        pred = model.predict(input_data)
        
        future_preds.append(pred)
        history.append(pred)
        
    return pd.DataFrame({'date': future_dates, 'forecast': future_preds})

# ---------- 7️⃣ الرسوم والتحليلات ----------
def show_metrics(f_df):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>${round(f_df['forecast'].mean(),2)}</div><div class='metric-label'>Average Forecast</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>${round(f_df['forecast'].max(),2)}</div><div class='metric-label'>Peak Sales</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>${round(f_df['forecast'].min(),2)}</div><div class='metric-label'>Lowest Sales</div></div>", unsafe_allow_html=True)

def plot_chart(h_df, f_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_df['date'].tail(20), y=h_df['sales'].tail(20), name='Recent Sales', line=dict(color="#334155", width=3)))
    fig.add_trace(go.Scatter(x=f_df['date'], y=f_df['forecast'], name='AI Forecast', line=dict(color="#2563eb", width=4, dash='dot'), marker=dict(size=8, symbol='star')))
    fig.update_layout(template='plotly_white', height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ================== MAIN APP ==================
def main():
    apply_css()
    render_header()

    st.subheader("📥 Predict Future Trends")
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Select Forecast Horizon", 7, 60, 30)
    with col2:
        last_val = st.number_input("Enter Current/Last Sales ($)", value=float(df['sales'].iloc[-1]))

    if st.button("🚀 Run AI Forecasting Model"):
        forecast_df = generate_forecast(model, df, feature_names, last_val, days)
        show_metrics(forecast_df)
        plot_chart(df, forecast_df)

if __name__ == "__main__":
    main()
