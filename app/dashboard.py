# ==============================
# Retail Sales Forecasting AI
# Developed by Eng. Goda Emad
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import plotly.graph_objects as go

# ================== Page Config ==================
st.set_page_config(page_title="Retail Sales Forecasting AI", layout="wide")

# ================== CSS Design ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e2e8f0, #f8fafc);
    font-family: 'Segoe UI', sans-serif;
}

.header-card {
    background: white;
    padding: 35px;
    border-radius: 22px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 35px;
}

.name-title {
    font-size: 44px;
    font-weight: 900;
    color: #0f172a;
}

.project-title {
    font-size: 28px;
    font-weight: 700;
    color: #2563eb;
}

.project-subtitle {
    font-size: 16px;
    color: #64748b;
    margin-top: 6px;
    margin-bottom: 15px;
}

.social-links a{
    text-decoration:none;
    margin: 0 10px;
    padding:10px 18px;
    border-radius:8px;
    font-weight:600;
    color:white;
}

.linkedin{background:#0A66C2;}
.github{background:#24292e;}

.metric-card {
    background:white;
    padding:25px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
}

.metric-value{
    font-size:36px;
    font-weight:700;
    color:#2563eb;
}

.metric-label{
    color:#64748b;
    font-size:15px;
}

.stButton>button{
    background:#2563eb;
    color:white;
    border-radius:10px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown("""
<div class='header-card'>
    <div class='name-title'>Eng. Goda Emad</div>
    <div class='project-title'>Smart Retail Sales Forecasting AI</div>
    <div class='project-subtitle'>
        Interactive Machine Learning Dashboard for Predicting Future Retail Sales
    </div>
    <div class='social-links'>
        <a href='https://www.linkedin.com/in/goda-emad/' target='_blank' class='linkedin'>LinkedIn</a>
        <a href='https://github.com/Goda-Emad' target='_blank' class='github'>GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Load Data & Model ==================
data = pd.read_csv("daily_sales_ready.csv")
data['date'] = pd.to_datetime(data['date'])

model = joblib.load("model.pkl")

# ================== User Input ==================
st.subheader("📥 Enter Inputs For Forecast")

col1, col2 = st.columns(2)

with col1:
    days = st.slider("Forecast Days", 7, 60, 30)

with col2:
    last_sales = st.number_input("Last Known Sales", value=float(data['sales'].iloc[-1]))

if st.button("🚀 Predict Future Sales"):

    future_preds = []
    current_sales = last_sales
    last_date = data['date'].max()

    for i in range(days):
        next_date = last_date + timedelta(days=i+1)

        features = np.array([[
            next_date.day,
            next_date.month,
            next_date.weekday(),
            current_sales
        ]])

        pred = model.predict(features)[0]
        future_preds.append((next_date, pred))
        current_sales = pred

    forecast_df = pd.DataFrame(future_preds, columns=["date", "forecast"])

    # ================== Metrics ==================
    st.subheader("📊 Forecast Insights")
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{round(forecast_df['forecast'].mean(),2)}</div>
        <div class='metric-label'>Average Forecast</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{round(forecast_df['forecast'].max(),2)}</div>
        <div class='metric-label'>Peak Sales</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{round(forecast_df['forecast'].min(),2)}</div>
        <div class='metric-label'>Lowest Sales</div>
    </div>
    """, unsafe_allow_html=True)

    # ================== Chart ==================
    st.subheader("📈 Sales Forecast Chart")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['sales'],
        mode='lines',
        name='Historical Sales'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast'
    ))

    st.plotly_chart(fig, use_container_width=True)

