# ==============================
# Retail AI Pro Dashboard
# ==============================

import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go

# ==============================
# 1️⃣ إعداد المسارات النسبية
# ==============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DAILY_SALES_PATH       = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
MODEL_PATH             = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
SCALER_PATH            = os.path.join(CURRENT_DIR, "scaler.pkl")
FEATURE_NAMES_PATH     = os.path.join(CURRENT_DIR, "feature_names.pkl")
FEATURE_IMPORTANCE_PATH= os.path.join(CURRENT_DIR, "feature_importance.pkl")
FORECAST_PATH          = os.path.join(CURRENT_DIR, "forecast_results.parquet")
MODEL_METRICS_PATH     = os.path.join(CURRENT_DIR, "model_metrics.pkl")
PRODUCT_ANALYTICS_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ==============================
# 2️⃣ التحقق من وجود الملفات
# ==============================
required_files = [
    DAILY_SALES_PATH, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    FEATURE_IMPORTANCE_PATH, FORECAST_PATH, MODEL_METRICS_PATH, PRODUCT_ANALYTICS_PATH
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

required_cols = ['date', 'sales']
for col in required_cols:
    if col not in daily_sales.columns:
        st.error(f"🚨 العمود '{col}' غير موجود في daily_sales_ready.parquet!")
        st.stop()

daily_sales['date'] = pd.to_datetime(daily_sales['date'])

# ==============================
# 4️⃣ تحميل الموديل والـ Scaler والميزات
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_NAMES_PATH)
feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)

# ==============================
# 5️⃣ Streamlit Page Setup
# ==============================
st.set_page_config(page_title="Retail AI Pro | Dashboard", layout="wide")
st.title("Retail AI Pro | Dashboard")

# ==============================
# 6️⃣ Forecast vs Actual Plot
# ==============================
forecast_df = pd.read_parquet(FORECAST_PATH)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_sales['date'], y=daily_sales['sales'],
    mode='lines', name='Actual', line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
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
# 7️⃣ عرض أهم المقاييس
# ==============================
model_metrics = joblib.load(MODEL_METRICS_PATH)
st.subheader("Model Performance")
st.write(model_metrics)

# ==============================
# 8️⃣ عرض تحليل المنتجات
# ==============================
product_analytics = pd.read_parquet(PRODUCT_ANALYTICS_PATH)
st.subheader("Top Products Analysis")
st.dataframe(product_analytics.sort_values('revenue', ascending=False).head(10))
