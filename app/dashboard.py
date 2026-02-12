import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ================== Page Config ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

st.title("🛒 Retail AI Sales Forecast Dashboard")
st.markdown("### Powered by CatBoost & Advanced Feature Engineering")

# ================== Paths ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DAILY_SALES_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready_10features.parquet")
FORECAST_PATH = os.path.join(CURRENT_DIR, "forecast_results.parquet")
METRICS_PATH = os.path.join(CURRENT_DIR, "model_metrics.pkl")
IMPORTANCE_PATH = os.path.join(CURRENT_DIR, "feature_importance_10features.pkl")
PRODUCT_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ================== Load Data ==================
daily_sales = pd.read_parquet(DAILY_SALES_PATH)
forecast_df = pd.read_parquet(FORECAST_PATH)
metrics = joblib.load(METRICS_PATH)
importance_df = joblib.load(IMPORTANCE_PATH)
product_df = pd.read_parquet(PRODUCT_PATH)

# ================== Sales vs Forecast ==================
st.subheader("📈 Actual Sales vs Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily_sales['date'],
    y=daily_sales['sales'],
    mode='lines',
    name='Actual Sales'
))

fig.add_trace(go.Scatter(
    x=forecast_df['date'],
    y=forecast_df['forecast'],
    mode='lines',
    name='Forecast'
))

st.plotly_chart(fig, use_container_width=True)

# ================== Model Metrics ==================
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:.2f}")
col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
col3.metric("R² Score", f"{metrics['R2']:.4f}")

# ================== Feature Importance ==================
st.subheader("🔑 Feature Importance")

importance_df = importance_df.sort_values("Importance", ascending=True)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance"
)

st.plotly_chart(fig_imp, use_container_width=True)

# ================== Product Analytics ==================
st.subheader("📦 Product Analytics")

top_products = product_df.sort_values("total_sales", ascending=False).head(10)

fig_prod = px.bar(
    top_products,
    x="total_sales",
    y="product",
    orientation="h",
    text="total_sales"
)

st.plotly_chart(fig_prod, use_container_width=True)

st.dataframe(top_products)
