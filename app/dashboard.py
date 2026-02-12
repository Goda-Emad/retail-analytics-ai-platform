# ==============================
# Retail AI Pro Dashboard - Fixed Version
# ==============================

import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go

# ==============================
# 1️⃣ إعداد المسارات
# ==============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DAILY_SALES_PATH       = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
MODEL_PATH             = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
SCALER_PATH            = os.path.join(CURRENT_DIR, "scaler.pkl")
FEATURE_NAMES_PATH     = os.path.join(CURRENT_DIR, "feature_names.pkl")
FEATURE_IMPORTANCE_PATH= os.path.join(CURRENT_DIR, "feature_importance.pkl")
FORECAST_PATH          = os.path.join(CURRENT_DIR, "Sales_Forecast_Feb_March_2026.xlsx")
MODEL_METRICS_PATH     = os.path.join(CURRENT_DIR, "model_metrics.pkl")
PRODUCT_ANALYTICS_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ==============================
# 2️⃣ التحقق من الملفات
# ==============================
for file in [DAILY_SALES_PATH, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
             FEATURE_IMPORTANCE_PATH, FORECAST_PATH, MODEL_METRICS_PATH, PRODUCT_ANALYTICS_PATH]:
    if not os.path.exists(file):
        st.error(f"🚨 الملف مفقود: {os.path.basename(file)}")
        st.stop()

# ==============================
# 3️⃣ قراءة البيانات
# ==============================
daily_sales = pd.read_parquet(DAILY_SALES_PATH)
daily_sales.columns = [str(col).lower() for col in daily_sales.columns]
daily_sales['date'] = pd.to_datetime(daily_sales['date'])

forecast_df = pd.read_excel(FORECAST_PATH)
forecast_df.columns = [str(col).lower() for col in forecast_df.columns]

product_analytics = pd.read_parquet(PRODUCT_ANALYTICS_PATH)
product_analytics.columns = [str(col).lower() for col in product_analytics.columns]

# ==============================
# 4️⃣ تحميل الموديل والـ Scaler والميزات
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_NAMES_PATH)
feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
model_metrics = joblib.load(MODEL_METRICS_PATH)

# ==============================
# 5️⃣ Page Setup
# ==============================
st.set_page_config(page_title="Retail AI Pro | Dashboard", layout="wide")
st.title("Retail AI Pro | Dashboard")

# ==============================
# 6️⃣ Forecast vs Actual
# ==============================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_sales['date'], y=daily_sales['sales'],
    mode='lines', name='Actual', line=dict(color='blue')
))
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
# 7️⃣ Model Performance
# ==============================
st.subheader("📈 Model Performance")
st.write(model_metrics)

# ==============================
# 8️⃣ Top Products
# ==============================
st.subheader("📦 Top Products Analysis")
# العمود الصحيح هو total_price بدل revenue
if 'total_price' in product_analytics.columns:
    st.dataframe(product_analytics.sort_values('total_price', ascending=False).head(10))
else:
    st.warning("⚠️ العمود 'total_price' غير موجود في product_analytics.parquet")
