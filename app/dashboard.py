# ========================= Imports =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from catboost import CatBoostRegressor
import os

st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ========================= Paths =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "daily_sales_ready_10features.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "catboost_sales_model_10features.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names_10features.pkl")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance_10features.pkl")

# ========================= Load Files =========================
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)

@st.cache_data
def load_importance():
    return joblib.load(IMPORTANCE_PATH)

daily_sales = load_data()
model = load_model()
feature_names = load_features()
importance_df = load_importance()

# ========================= Title =========================
st.title("📊 Retail Sales Forecast Dashboard")
st.subheader("Built by Eng. Goda Emad")

# ========================= Sales Chart =========================
st.header("📈 Historical Sales")

daily_sales['date'] = pd.to_datetime(daily_sales['date'])
fig = px.line(daily_sales, x='date', y='sales', title="Daily Sales Over Time")
st.plotly_chart(fig, use_container_width=True)

# ========================= Prepare Features =========================
X = daily_sales[feature_names].copy()
y_pred = model.predict(X)

daily_sales['prediction'] = y_pred

# ========================= Prediction Chart =========================
st.header("🔮 Model Predictions vs Actual")

fig2 = px.line(daily_sales, x='date', y=['sales', 'prediction'],
               title="Actual vs Predicted Sales")
st.plotly_chart(fig2, use_container_width=True)

# ========================= Feature Importance =========================
st.header("⭐ Feature Importance (Top Drivers)")

importance_df = importance_df.sort_values(by="Importance", ascending=False)

fig3 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation='h',
    title="Feature Importance from CatBoost"
)
st.plotly_chart(fig3, use_container_width=True)

# ========================= Metrics =========================
st.header("📌 Dataset Info")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(daily_sales))
col2.metric("Start Date", str(daily_sales['date'].min().date()))
col3.metric("End Date", str(daily_sales['date'].max().date()))
