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

# ---------- 4️⃣ إعداد المسارات الذكية ----------
# الحصول على المسار الرئيسي للمشروع (Root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "daily_sales_ready.csv"
MODEL_PATH = BASE_DIR / "model" / "catboost_sales_model.pkl"

# ---------- 5️⃣ تحميل البيانات والموديل ----------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"❌ File not found at: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

# ---------- 6️⃣ منطق التوقع (Recursive Forecast) ----------
def recursive_forecast(model, last_date, last_sales, days):
    preds = []
    current_sales = last_sales

    for i in range(days):
        next_date = last_date + timedelta(days=i+1)
        # مصفوفة المدخلات للموديل (يجب أن تطابق ما تم في Colab)
        features = np.array([[
            next_date.day,
            next_date.month,
            next_date.weekday(),
            current_sales
        ]])
        pred = model.predict(features)[0]
        preds.append((next_date, pred))
        current_sales = pred

    return pd.DataFrame(preds, columns=["date", "forecast"])

# ---------- 7️⃣ عرض الإحصائيات (Metrics) ----------
def show_metrics(forecast_df):
    c1, c2, c3 = st.columns(3)
    
    avg_val = round(forecast_df['forecast'].mean(), 2)
    max_val = round(forecast_df['forecast'].max(), 2)
    min_val = round(forecast_df['forecast'].min(), 2)

    c1.markdown(f"<div class='metric-card'><div class='metric-value'>${avg_val}</div><div class='metric-label'>Average Forecast</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>${max_val}</div><div class='metric-label'>Peak Sales</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>${min_val}</div><div class='metric-label'>Lowest Sales</div></div>", unsafe_allow_html=True)

# ---------- 8️⃣ الرسم البياني التفاعلي ----------
def plot_chart(history_df, forecast_df):
    fig = go.Figure()

    # التاريخ الفعلي (آخر 30 يوم لزيادة الوضوح)
    fig.add_trace(go.Scatter(
        x=history_df['date'].tail(30),
        y=history_df['sales'].tail(30),
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color="#334155", width=3)
    ))

    # التوقع المستقبلي
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color="#2563eb", width=4, dash='dot'),
        marker=dict(size=8, symbol='star')
    ))

    fig.update_layout(
        template='plotly_white',
        height=600,
        hovermode="x unified",
        title_text="Sales Trend Analysis & Future Prediction",
        xaxis_title="Date",
        yaxis_title="Sales Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ================== 🚀 التطبيق الرئيسي ==================
def main():
    apply_css()
    render_header()

    # تحميل الموارد
    df = load_data()
    model = load_model()

    st.markdown("### 📥 Setup Your Forecast")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        days = st.slider("Select Forecast Horizon (Days)", 7, 60, 30)
    with col_in2:
        last_val = float(df['sales'].iloc[-1])
        last_sales = st.number_input("Last Known Sales Value ($)", value=last_val)

    st.markdown("---")
    
    # تنفيذ التوقع عند الضغط على الزر
    if st.button("🚀 Generate AI Prediction"):
        with st.spinner('AI is calculating future trends...'):
            forecast_result = recursive_forecast(
                model, 
                df['date'].max(), 
                last_sales, 
                days
            )

            st.markdown("### 📊 Performance Insights")
            show_metrics(forecast_result)

            st.markdown("### 📈 Visual Analytics")
            plot_chart(df, forecast_result)
            
            # خيار تحميل البيانات
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=forecast_result.to_csv(index=False),
                file_name=f"forecast_goda_emad_{days}days.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
