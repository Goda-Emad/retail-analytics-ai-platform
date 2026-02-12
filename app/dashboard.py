# ==============================
# Retail AI Pro | Professional Dashboard
# ==============================

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# 1️⃣ إعداد المسارات
# ==============================
DAILY_SALES_PATH       = "daily_sales_ready_10features.parquet"
MODEL_PATH             = "catboost_sales_model_10features.pkl"
SCALER_PATH            = "scaler_10features.pkl"
FEATURE_NAMES_PATH     = "feature_names_10features.pkl"
FORECAST_PATH          = "forecast_results.parquet"
MODEL_METRICS_PATH     = "model_metrics.pkl"
PRODUCT_ANALYTICS_PATH = "product_analytics.parquet"
SALES_FORECAST_XLSX    = "Sales_Forecast_Feb_March_2026.xlsx"

# ==============================
# 2️⃣ قراءة البيانات الأساسية
# ==============================
daily_sales = pd.read_parquet(DAILY_SALES_PATH)
daily_sales['date'] = pd.to_datetime(daily_sales['date'])

forecast_df = pd.read_parquet(FORECAST_PATH)
product_analytics = pd.read_parquet(PRODUCT_ANALYTICS_PATH)
model_metrics = joblib.load(MODEL_METRICS_PATH)

# ==============================
# 3️⃣ إعداد واجهة Streamlit
# ==============================
st.set_page_config(page_title="Retail AI Pro | Dashboard", layout="wide")
st.title("📊 Retail AI Pro | Dashboard")

# فلتر التاريخ
st.sidebar.subheader("تصفية حسب التاريخ")
start_date = st.sidebar.date_input("Start Date", daily_sales['date'].min())
end_date   = st.sidebar.date_input("End Date", daily_sales['date'].max())
mask = (daily_sales['date'] >= pd.to_datetime(start_date)) & (daily_sales['date'] <= pd.to_datetime(end_date))
daily_sales_filtered = daily_sales.loc[mask]
forecast_filtered = forecast_df[(forecast_df['date'] >= pd.to_datetime(start_date)) & 
                                (forecast_df['date'] <= pd.to_datetime(end_date))]

# ==============================
# 4️⃣ مخطط المبيعات الفعلية مقابل التوقعات
# ==============================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_sales_filtered['date'],
    y=daily_sales_filtered['sales'],
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=forecast_filtered['date'],
    y=forecast_filtered['predicted_sales'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='orange')
))

fig.update_layout(
    title="📈 Actual vs Forecast Sales",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# 5️⃣ عرض مقاييس الموديل
# ==============================
st.subheader("🔹 Model Performance")
st.write(model_metrics)

# ==============================
# 6️⃣ عرض أفضل المنتجات
# ==============================
st.subheader("🛒 Top Products Analysis")
product_analytics_sorted = product_analytics.sort_values('Total_Price', ascending=False)
st.dataframe(product_analytics_sorted.head(10))

# ==============================
# 7️⃣ تحميل التوقعات المستقبلية
# ==============================
st.subheader("📅 Future Sales Forecast")
st.download_button(
    label="Download Sales Forecast (Feb-Mar 2026)",
    data=open(SALES_FORECAST_XLSX, "rb"),
    file_name="Sales_Forecast_Feb_March_2026.xlsx"
)

