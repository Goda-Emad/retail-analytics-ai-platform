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
import plotly.graph_objects as go


# ---------- Page Config ----------
st.set_page_config(page_title="Retail AI Forecast", layout="wide")


# ---------- Styling ----------
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
        border-radius:10px;height:55px;font-size:18px;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------- Header ----------
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


# ---------- Load Resources ----------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.csv")

    if not os.path.exists(DATA_PATH):
        st.error(f"❌ File not found: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File not found: {MODEL_PATH}")
        st.stop()

    return joblib.load(MODEL_PATH)


# ---------- Forecast Logic ----------
def recursive_forecast(model, last_date, last_sales, days):
    preds = []
    current_sales = last_sales

    for i in range(days):
        next_date = last_date + timedelta(days=i+1)
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


# ---------- Metrics ----------
def show_metrics(forecast_df):
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{round(forecast_df['forecast'].mean(),2)}</div>
        <div class='metric-label'>Average Forecast</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{round(forecast_df['forecast'].max(),2)}</div>
        <div class='metric-label'>Peak Sales</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{round(forecast_df['forecast'].min(),2)}</div>
        <div class='metric-label'>Lowest Sales</div>
    </div>""", unsafe_allow_html=True)


# ---------- Chart ----------
def plot_chart(history_df, forecast_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['sales'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color="#334155", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color="#2563eb", width=4, dash='dot'),
        marker=dict(size=8, symbol='star')
    ))

    fig.update_layout(
        template='plotly_white',
        height=600,
        hovermode="x unified",
        title_text="Historical Sales & Forecast",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


# ================== MAIN APP ==================
def main():
    apply_css()
    render_header()

    df = load_data()
    model = load_model()

    st.subheader("📥 Enter Forecast Inputs")

    col1, col2 = st.columns(2)

    with col1:
        days = st.slider("Forecast Days", 7, 60, 30)

    with col2:
        last_sales = st.number_input(
            "Last Known Sales",
            value=float(df['sales'].iloc[-1])
        )

    if st.button("🚀 Predict Future Sales"):
        forecast_df = recursive_forecast(
            model,
            df['date'].max(),
            last_sales,
            days
        )

        st.subheader("📊 Forecast Insights")
        show_metrics(forecast_df)

        st.subheader("📈 Forecast Chart")
        plot_chart(df, forecast_df)


if __name__ == "__main__":
    main()
