import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from catboost import CatBoostRegressor
import joblib
import os

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro v9 | Real Forecast", layout="wide")

# ================== Theme ==================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light 🌞"

def toggle_theme():
    st.session_state.theme_mode = "Dark 🌙" if st.session_state.theme_mode=="Light 🌞" else "Light 🌞"

st.button("🌗 Toggle Theme", on_click=toggle_theme)
theme_mode = st.session_state.theme_mode

if theme_mode == "Dark 🌙":
    bg_color = "#0f172a"
    text_color = "#f1f5f9"
    accent_color = "#3b82f6"
else:
    bg_color = "#f8fafc"
    text_color = "#1e293b"
    accent_color = "#2563eb"

# ================== Background & Header ==================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("../images/bg_retail_1.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='display:flex; align-items:center; padding:20px; position:fixed; width:100%; z-index:200;'>
    <img src='../images/retail_ai_pro_logo.webp' width='100'>
    <div style='margin-left:15px;'>
        <h1 style='margin:0; color:{accent_color};'>Retail AI Pro</h1>
        <h3 style='margin:0; color:{text_color};'>Built by Eng. Goda Emad</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Paths ==================
MODEL_PATH = "app/catboost_sales_model.pkl"
FEATURES_PATH = "app/feature_names.pkl"
DATA_PATH = "app/daily_sales_ready.parquet"

# ================== Load Model ==================
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    model: CatBoostRegressor = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
else:
    st.error("⚠️ Model files not found! Make sure 'catboost_sales_model.pkl' and 'feature_names.pkl' exist in the app folder.")
    st.stop()

# ================== Load Real Data ==================
if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)
else:
    st.error("⚠️ Data file not found! Make sure 'daily_sales_ready.parquet' exists in the app folder.")
    st.stop()

df = df.sort_values("InvoiceDate")
sales_hist = df.set_index("InvoiceDate")["TotalAmount"]

# ================== Feature Engineering ==================
def get_cyclical_features(date):
    day_sin = np.sin(2*np.pi*date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(date.isocalendar().week % 52)/52)
    month_sin = np.sin(2*np.pi*date.month/12)
    return day_sin, week_sin, month_sin

def get_lagged_features(series, lags=[1,7]):
    df_lag = pd.DataFrame()
    for lag in lags:
        if len(series) >= lag:
            df_lag[f"lag_{lag}"] = series.shift(lag)
        else:
            df_lag[f"lag_{lag}"] = 0
    return df_lag

# ================== Generate Forecast ==================
def generate_forecast(hist_series, horizon, scenario="Realistic", noise_level=0.03):
    forecast_values = []
    hist = hist_series.copy()

    for i in range(horizon):
        date = hist.index[-1] + pd.Timedelta(days=1)
        day_sin, week_sin, month_sin = get_cyclical_features(date)
        lag_features = get_lagged_features(hist, lags=[1,7])
        latest_lags = lag_features.iloc[-1].values
        X = np.array([day_sin, week_sin, month_sin] + list(latest_lags)).reshape(1,-1)

        X_df = pd.DataFrame(X, columns=feature_names)
        pred = model.predict(X_df)[0]

        if scenario=="Optimistic (+15%)": pred *= 1.15
        elif scenario=="Pessimistic (-15%)": pred *= 0.85

        pred = pred * (1 + np.random.normal(0, noise_level))
        forecast_values.append(pred)
        hist.loc[date] = pred

    return np.array(forecast_values)

# ================== Sidebar Controls ==================
st.sidebar.header("Forecast Controls")
scenario = st.sidebar.selectbox("Market Scenario", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 30, 21)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.1, 0.03)
run_btn = st.sidebar.button("🚀 Run AI Forecast")

# ================== Run Forecast ==================
if run_btn:
    forecast = generate_forecast(sales_hist, horizon=horizon, scenario=scenario, noise_level=noise_level)
    future_dates = [sales_hist.index[-1] + timedelta(days=i+1) for i in range(horizon)]
    upper = forecast * 1.05
    lower = forecast * 0.95

    # ===== Forecast Chart =====
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_hist.index, y=sales_hist.values, mode='lines+markers', name="History", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name="Forecast", line=dict(color=accent_color, width=3)))
    fig.add_trace(go.Scatter(x=future_dates+future_dates[::-1], y=list(upper)+list(lower[::-1]),
                             fill='toself', fillcolor='rgba(59,130,246,0.15)', line=dict(color='rgba(0,0,0,0)'), hoverinfo="skip", name="Error Margin"))
    fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=text_color,
                      xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    # ===== Forecast Table & Download =====
    df_forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast,
        "Lower": lower,
        "Upper": upper
    })
    st.subheader("📊 Forecast Table")
    st.dataframe(df_forecast.style.format({"Forecast":"${:,.0f}","Lower":"${:,.0f}","Upper":"${:,.0f}"}))
    st.download_button("📥 Download Forecast CSV", df_forecast.to_csv(index=False), "forecast.csv", "text/csv")

    # ===== Backtest =====
    backtest_days = min(30, len(sales_hist)-7)
    actual_back = sales_hist[-backtest_days:]
    preds_back = []

    hist_temp = sales_hist[:-backtest_days].copy()
    for date in actual_back.index:
        day_sin, week_sin, month_sin = get_cyclical_features(date)
        lag_features = get_lagged_features(hist_temp, lags=[1,7])
        latest_lags = lag_features.iloc[-1].values
        X = np.array([day_sin, week_sin, month_sin] + list(latest_lags)).reshape(1,-1)
        X_df = pd.DataFrame(X, columns=feature_names)
        pred = model.predict(X_df)[0]
        preds_back.append(pred)
        hist_temp.loc[date] = actual_back.loc[date]

    preds_back = np.array(preds_back)
    mape = np.mean(np.abs((actual_back - preds_back)/actual_back))*100
    confidence = max(0, 100 - mape)

    st.markdown(f"**Backtest MAPE:** {mape:.2f}% | **Confidence:** {confidence:.2f}%")
