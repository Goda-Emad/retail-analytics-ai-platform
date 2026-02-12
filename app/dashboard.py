# ==============================
# Retail AI Pro | Dashboard
# ==============================

import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# 1️⃣ إعداد المسارات النسبية
# ==============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DAILY_SALES_PATH       = os.path.join(CURRENT_DIR, "daily_sales_ready_10features.parquet")
MODEL_PATH             = os.path.join(CURRENT_DIR, "catboost_sales_model_10features.pkl")
SCALER_PATH            = os.path.join(CURRENT_DIR, "scaler_10features.pkl")
FEATURE_NAMES_PATH     = os.path.join(CURRENT_DIR, "feature_names_10features.pkl")
FEATURE_IMPORTANCE_PATH= os.path.join(CURRENT_DIR, "feature_importance.pkl")
FORECAST_PATH          = os.path.join(CURRENT_DIR, "forecast_results.parquet")
MODEL_METRICS_PATH     = os.path.join(CURRENT_DIR, "model_metrics.pkl")
PRODUCT_ANALYTICS_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")
EXCEL_FORECAST_PATH    = os.path.join(CURRENT_DIR, "Sales_Forecast_Feb_March_2026.xlsx")

# ==============================
# 2️⃣ التحقق من وجود الملفات
# ==============================
required_files = [
    DAILY_SALES_PATH, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    FEATURE_IMPORTANCE_PATH, FORECAST_PATH, MODEL_METRICS_PATH, PRODUCT_ANALYTICS_PATH,
    EXCEL_FORECAST_PATH
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"🚨 الملف مفقود: {os.path.basename(file)}")
        st.stop()

# ==============================
# 3️⃣ قراءة البيانات
# ==============================
daily_sales = pd.read_parquet(DAILY_SALES_PATH)
daily_sales.columns = [str(c).lower().strip() for c in daily_sales.columns]
daily_sales['date'] = pd.to_datetime(daily_sales['date'])

forecast_df = pd.read_parquet(FORECAST_PATH)
forecast_df.columns = [str(c).lower().strip() for c in forecast_df.columns]
forecast_df['date'] = pd.to_datetime(forecast_df['date'])

excel_forecast = pd.read_excel(EXCEL_FORECAST_PATH)
excel_forecast.columns = [str(c).lower().strip() for c in excel_forecast.columns]
excel_forecast['date'] = pd.to_datetime(excel_forecast['date'])

product_analytics = pd.read_parquet(PRODUCT_ANALYTICS_PATH)
product_analytics.columns = [str(c).lower().strip() for c in product_analytics.columns]

# ==============================
# 4️⃣ تحميل الموديل، Scaler، وميزات الموديل
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
model_metrics = joblib.load(MODEL_METRICS_PATH)

# ==============================
# 5️⃣ إعداد صفحة Streamlit
# ==============================
st.set_page_config(page_title="Retail AI Pro | Dashboard", layout="wide")
st.title("Retail AI Pro | Dashboard")

# ==============================
# 6️⃣ Chart: Actual vs Forecast
# ==============================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_sales['date'], y=daily_sales['sales'],
    mode='lines', name='Actual Sales', line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=forecast_df['date'], y=forecast_df['predicted_sales'],
    mode='lines', name='Forecast (Model)', line=dict(color='orange')
))
fig.add_trace(go.Scatter(
    x=excel_forecast['date'], y=excel_forecast['predicted_sales'],
    mode='lines', name='Forecast Feb-Mar 2026', line=dict(color='green', dash='dash')
))

fig.update_layout(
    title="📊 Actual vs Forecast Sales",
    xaxis_title="Date",
    yaxis_title="Sales",
    legend=dict(x=0, y=1),
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# 7️⃣ Model Metrics
# ==============================
st.subheader("🏆 Model Performance")
st.write(model_metrics)

# ==============================
# 8️⃣ Feature Importance
# ==============================
st.subheader("🔑 Feature Importance")
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)
st.dataframe(importance_df)

# ==============================
# 9️⃣ Top Products Analysis
# ==============================
st.subheader("🛍️ Top Products by Revenue")
if 'total_price' in product_analytics.columns:
    product_analytics_sorted = product_analytics.sort_values('total_price', ascending=False)
else:
    product_analytics_sorted = product_analytics
st.dataframe(product_analytics_sorted.head(10))

# ==============================
# 1️⃣0️⃣ Footer
# ==============================
st.markdown("---")
st.markdown("Built by Eng. Goda Emad | Retail AI Pro")

