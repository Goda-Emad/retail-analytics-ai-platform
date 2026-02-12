# ==========================
# Retail AI Pro - Dashboard
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ================== Paths ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
FORECAST_PATH = os.path.join(CURRENT_DIR, "Sales_Forecast_Feb_March_2026.xlsx")
PRODUCT_ANALYTICS_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ================== Load Assets ==================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURES_PATH)

# Load historical & forecast data
daily_sales = pd.read_parquet(os.path.join(CURRENT_DIR, "daily_sales_ready.parquet"))
forecast_df = pd.read_excel(FORECAST_PATH)
product_analytics = pd.read_parquet(PRODUCT_ANALYTICS_PATH)

# ================== Streamlit Page Setup ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

st.title("📊 Retail AI Pro Dashboard")
st.markdown("**End-to-End Sales Forecasting & Analysis Platform**")

# ================== Tabs ==================
tab1, tab2, tab3 = st.tabs(["Historical vs Forecast", "Product Analytics", "Metrics & Insights"])

# ================== Tab 1: Forecast vs Actual ==================
with tab1:
    st.subheader("Actual Sales vs Forecasted Sales")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['sales'],
                             mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
                             mode='lines', name='Forecast', line=dict(color='orange')))
    fig.update_layout(title="Forecast vs Actual Sales",
                      xaxis_title="Date", yaxis_title="Sales",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ================== Tab 2: Product Analytics ==================
with tab2:
    st.subheader("Product Analytics Overview")
    st.dataframe(product_analytics)

# ================== Tab 3: Metrics & Insights ==================
with tab3:
    st.subheader("Model Performance & Key Insights")
    metrics_text = f"""
    ✅ Model: CatBoost Regressor
    ✅ Total Historical Days: {len(daily_sales)}
    ✅ Forecast Horizon: {forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}
    ✅ Stable WMAPE (Historical Folds Average): 47.87%
    
    🔹 Key Features (SHAP Analysis):
      - rolling_mean_14
      - dayofweek_sin
      - was_closed_yesterday
    
    💡 Insights:
    - Latest folds show improved accuracy (~80% on recent data)
    - Recursive forecasting applied for next 30 days
    """
    st.markdown(metrics_text)
