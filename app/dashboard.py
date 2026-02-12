# ============================== #
#        Retail AI Pro          #
#   CatBoost Production App     #
# ============================== #

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="Retail AI Pro",
    page_icon="📊",
    layout="wide"
)

# ==============================
# Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(BASE_DIR, "daily_sales_ready.parquet")
PRODUCT_PATH = os.path.join(BASE_DIR, "product_analytics.parquet")

# ==============================
# Load Resources
# ==============================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names

@st.cache_data
def load_data():
    sales = pd.read_parquet(DATA_PATH)
    sales = sales.reset_index()
    sales[sales.columns[0]] = pd.to_datetime(sales[sales.columns[0]])
    sales = sales.set_index(sales.columns[0])
    sales = sales.sort_index()

    products = pd.read_parquet(PRODUCT_PATH)
    if "Description" in products.columns:
        products = products.rename(columns={"Description": "Product"})

    return sales, products

model, feature_names = load_model()
sales_df, product_df = load_data()

# ==============================
# Feature Engineering
# ==============================

def build_features(history: pd.Series, next_date):

    history = history.ffill().fillna(0)

    day_sin = np.sin(2*np.pi*next_date.dayofweek/7)
    day_cos = np.cos(2*np.pi*next_date.dayofweek/7)

    week = next_date.isocalendar().week % 52
    week_sin = np.sin(2*np.pi*week/52)
    week_cos = np.cos(2*np.pi*week/52)

    month_sin = np.sin(2*np.pi*next_date.month/12)
    month_cos = np.cos(2*np.pi*next_date.month/12)

    lag_1 = history.iloc[-1]
    lag_7 = history.iloc[-7] if len(history) >= 7 else history.mean()
    lag_30 = history.iloc[-30] if len(history) >= 30 else history.mean()

    rolling_7 = history[-7:].mean() if len(history) >= 7 else history.mean()
    rolling_30 = history[-30:].mean() if len(history) >= 30 else history.mean()

    features = {
        "day_sin": day_sin,
        "day_cos": day_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_month_end": int(next_date.is_month_end),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_30": lag_30,
        "rolling_mean_7": rolling_7,
        "rolling_mean_30": rolling_30
    }

    X = pd.DataFrame([features])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].astype(float)

    return X

# ==============================
# Forecast Engine
# ==============================

def forecast_series(series, horizon, scenario="واقعي"):

    history = series.copy()
    predictions = []

    for _ in range(horizon):

        next_date = history.index[-1] + timedelta(days=1)

        X = build_features(history, next_date)

        pred = model.predict(X)[0]

        if scenario == "متفائل (+15%)":
            pred *= 1.15
        elif scenario == "متشائم (-15%)":
            pred *= 0.85

        pred = max(0, float(pred))

        predictions.append(pred)
        history.loc[next_date] = pred

    return np.array(predictions), history.index[-horizon:]

# ==============================
# Backtest (No Leakage)
# ==============================

def backtest(series, test_size=30):

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    preds, dates = forecast_series(train, test_size)

    mape = np.mean(np.abs((test.values - preds) / test.values)) * 100
    mae = mean_absolute_error(test.values, preds)
    rmse = np.sqrt(mean_squared_error(test.values, preds))

    return test, preds, dates, mape, mae, rmse

# ==============================
# Sidebar
# ==============================

st.sidebar.title("⚙ Control Panel")

scenario = st.sidebar.selectbox(
    "Market Scenario",
    ["واقعي", "متفائل (+15%)", "متشائم (-15%)"]
)

horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 14)

run_forecast = st.sidebar.button("🚀 Run Forecast")

# ==============================
# Tabs
# ==============================

tab1, tab2 = st.tabs(["🔮 Forecasting", "📊 Analytics & Backtest"])

# ==============================
# TAB 1
# ==============================

with tab1:

    st.title("Retail AI Pro")
    st.subheader("CatBoost Sales Forecasting Engine")

    if run_forecast:

        with st.spinner("Running AI Forecast..."):

            preds, future_dates = forecast_series(
                sales_df["Daily_Sales"],
                horizon,
                scenario
            )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sales_df.index,
            y=sales_df["Daily_Sales"],
            name="Historical",
            line=dict(color="gray")
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds,
            name="Forecast",
            line=dict(color="cyan", width=3)
        ))

        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        col1.metric("Total Forecast", f"${preds.sum():,.0f}")
        col2.metric("Daily Average", f"${preds.mean():,.0f}")

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": preds
        })

        st.download_button(
            "Download Forecast CSV",
            forecast_df.to_csv(index=False),
            "forecast.csv",
            "text/csv"
        )

# ==============================
# TAB 2
# ==============================

with tab2:

    st.subheader("Model Performance Backtest")

    test, preds, dates, mape, mae, rmse = backtest(
        sales_df["Daily_Sales"], 30
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=test.index,
        y=test.values,
        name="Actual",
        line=dict(color="green")
    ))

    fig2.add_trace(go.Scatter(
        x=dates,
        y=preds,
        name="Predicted",
        line=dict(color="blue")
    ))

    st.plotly_chart(fig2, use_container_width=True)

    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
