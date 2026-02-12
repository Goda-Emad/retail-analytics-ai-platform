# ==============================
# Retail AI Pro Dashboard (10 Features)
# ==============================

import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1️⃣ إعداد المسارات النسبية
# ==============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DAILY_SALES_PATH       = os.path.join(CURRENT_DIR, "daily_sales_ready_10features.parquet")
MODEL_PATH             = os.path.join(CURRENT_DIR, "catboost_sales_model_10features.pkl")
SCALER_PATH            = os.path.join(CURRENT_DIR, "scaler_10features.pkl")
FEATURE_NAMES_PATH     = os.path.join(CURRENT_DIR, "feature_names_10features.pkl")
FORECAST_PATH          = os.path.join(CURRENT_DIR, "forecast_results.parquet")
MODEL_METRICS_PATH     = os.path.join(CURRENT_DIR, "model_metrics.pkl")
PRODUCT_ANALYTICS_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ==============================
# 2️⃣ التحقق من وجود الملفات
# ==============================
required_files = [
    DAILY_SALES_PATH, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    FORECAST_PATH, MODEL_METRICS_PATH, PRODUCT_ANALYTICS_PATH
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"🚨 الملف مفقود: {os.path.basename(file)}")
        st.stop()

# ==============================
# 3️⃣ قراءة البيانات
# ==============================
daily_sales = pd.read_parquet(DAILY_SALES_PATH)
daily_sales.columns = [str(col).lower().strip() for col in daily_sales.columns]

required_cols = ['date', 'sales',
                 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
                 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14',
                 'is_weekend', 'was_closed_yesterday']

for col in required_cols:
    if col not in daily_sales.columns:
        st.error(f"🚨 العمود '{col}' غير موجود!")
        st.stop()

daily_sales['date'] = pd.to_datetime(daily_sales['date'])

# ==============================
# 4️⃣ تحميل الموديل والـ Scaler والميزات
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_NAMES_PATH)

# ==============================
# 5️⃣ Streamlit Page Setup
# ==============================
st.set_page_config(page_title="Retail AI Pro | Dashboard", layout="wide")
st.title("Retail AI Pro | Dashboard")

# ==============================
# 6️⃣ تحضير بيانات الـ Forecast
# ==============================
if os.path.exists(FORECAST_PATH):
    forecast_df = pd.read_parquet(FORECAST_PATH)
    forecast_df.columns = [str(col).lower().strip() for col in forecast_df.columns]
else:
    forecast_df = pd.DataFrame(columns=['date', 'predicted_sales'])

# ==============================
# 7️⃣ Plot: Actual vs Forecast
# ==============================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_sales['date'], y=daily_sales['sales'],
    mode='lines', name='Actual', line=dict(color='blue')
))
if not forecast_df.empty:
    fig.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['predicted_sales'],
        mode='lines', name='Forecast', line=dict(color='orange')
    ))

fig.update_layout(
    title="Actual vs Forecast Sales",
    xaxis_title="Date",
    yaxis_title="Sales",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 8️⃣ حساب Metrics
# ==============================
X = daily_sales[feature_order]
y = daily_sales['sales']

# Scale features
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred) ** 0.5
r2 = r2_score(y, y_pred)

st.subheader("Model Performance")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R2: {r2:.4f}")

# ==============================
# 9️⃣ عرض تحليل المنتجات
# ==============================
product_analytics = pd.read_parquet(PRODUCT_ANALYTICS_PATH)
st.subheader("Top Products Analysis")
st.dataframe(product_analytics.sort_values('total_price', ascending=False).head(10))
