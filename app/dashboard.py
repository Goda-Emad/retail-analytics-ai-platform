import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# ================== Page Config ==================
st.set_page_config(page_title="Retail AI Forecasting", layout="wide")

# ================== Premium CSS ==================
st.markdown("""
<style>
body {background-color: #f5f7fa;}
.main-title {
    font-size:48px;
    font-weight:800;
    color:#0f172a;
    text-align:center;
    margin-bottom:5px;
}
.sub-title {
    text-align:center;
    font-size:18px;
    color:#475569;
    margin-bottom:35px;
}
.card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 10px 25px rgba(0,0,0,0.08);
}
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

# ================== Load Files ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model_v2.pkl")
FEAT_PATH = os.path.join(BASE_DIR, "..", "model", "feature_names.pkl")

@st.cache_data
def load_essentials():
    df = pd.read_parquet(DATA_PATH)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_PATH)
    return df, model, feature_names

df, model, feature_names = load_essentials()

# ================== Header ==================
st.markdown("<div class='main-title'>Smart Retail Sales Forecasting AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Interactive AI Dashboard for Predicting Future Retail Sales</div>", unsafe_allow_html=True)

# ================== Inputs Row ==================
c1, c2, c3, c4 = st.columns([2,2,2,1])

default_lag1 = float(df['total_amount'].iloc[-1])
default_lag7 = float(df['total_amount'].iloc[-7])

with c1:
    lag1 = st.number_input("Yesterday's Sales ($)", value=default_lag1)

with c2:
    lag7 = st.number_input("Last Week's Sales ($)", value=default_lag7)

with c3:
    days = st.slider("Forecast Days", 7, 60, 30)

with c4:
    predict_btn = st.button("🔮 Predict")

# ================== Forecast Function ==================
def forecast(model, df_hist, feature_names, lag1, lag7, days):
    future_preds = []
    last_date = df_hist['InvoiceDate'].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days)
    history = list(df_hist['total_amount'].tail(30))
    history[-1] = lag1

    for d in future_dates:
        feats = {
            'day': d.day,
            'month': d.month,
            'dayofweek': d.dayofweek,
            'is_weekend': 1 if d.dayofweek in [4,5] else 0,
            'rolling_mean_7': pd.Series(history[-7:]).mean(),
            'lag_1': history[-1],
            'lag_7': history[-7]
        }
        X = [feats[f] for f in feature_names]
        pred = model.predict([X])[0]
        future_preds.append(pred)
        history.append(pred)

    return future_dates, future_preds

# ================== Run After Predict ==================
if predict_btn:

    dates, preds = forecast(model, df, feature_names, lag1, lag7, days)
    fdf = pd.DataFrame({'Date': dates, 'Sales': preds})

    peak = fdf['Sales'].idxmax()
    low = fdf['Sales'].idxmin()

    # ================== Chart ==================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['InvoiceDate'].tail(15),
        y=df['total_amount'].tail(15),
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='#0f172a', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=fdf['Date'],
        y=fdf['Sales'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#2563eb', width=4),
        marker=dict(size=9)
    ))

    fig.add_trace(go.Scatter(
        x=[fdf.loc[peak,'Date']],
        y=[fdf.loc[peak,'Sales']],
        mode='markers',
        marker=dict(size=14, color='red'),
        name='Peak'
    ))

    fig.add_trace(go.Scatter(
        x=[fdf.loc[low,'Date']],
        y=[fdf.loc[low,'Sales']],
        mode='markers',
        marker=dict(size=14, color='green'),
        name='Lowest'
    ))

    fig.update_layout(height=600, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ================== Metrics ==================
    st.markdown("### Key Insights")
    m1, m2, m3, m4 = st.columns(4)

    m1.markdown(f"<div class='metric-card'><div class='metric-value'>${preds[0]:,.0f}</div><div class='metric-label'>Tomorrow</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-value'>${sum(preds[:7]):,.0f}</div><div class='metric-label'>Next 7 Days</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-value'>${max(preds):,.0f}</div><div class='metric-label'>Peak Day</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div class='metric-value'>${min(preds):,.0f}</div><div class='metric-label'>Lowest Day</div></div>", unsafe_allow_html=True)

    # ================== Download ==================
    buffer = BytesIO()
    fdf.to_csv(buffer, index=False)
    st.download_button("📥 Download Forecast CSV", buffer.getvalue(), "forecast.csv")

# ================== Footer ==================
st.markdown("<br><center style='color:#64748b'>Developed by Eng. Goda Emad | CatBoost AI</center>", unsafe_allow_html=True)
