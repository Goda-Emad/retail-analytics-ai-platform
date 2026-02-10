import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# ================== Page Config ==================
st.set_page_config(page_title="Retail AI Forecasting | Eng. Goda Emad", layout="wide")

# ================== Premium CSS (Keeping your Original Design) ==================
st.markdown("""
<style>
.stApp { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #f0f4f8, #e0e7ff); overflow-x: hidden; }
.stApp::before {
    content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image: linear-gradient(rgba(37,99,235,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(37,99,235,0.05) 1px, transparent 1px);
    background-size: 40px 40px; z-index: 0; animation: moveLines 120s linear infinite;
}
@keyframes moveLines { 0% {background-position: 0 0;} 100% {background-position: 1000px 1000px;} }
.header-card { background: white; padding: 35px; border-radius: 22px; box-shadow: 0 15px 35px rgba(0,0,0,0.08); text-align: center; margin-bottom: 35px; }
.name-title { font-size: 42px; font-weight: 900; color: #0f172a; }
.project-title { font-size: 26px; font-weight: 700; color: #2563eb; }
.project-subtitle { font-size: 16px; color: #64748b; margin-top: 6px; }
.metric-card { background:white; padding:20px; border-radius:18px; text-align:center; box-shadow:0 8px 20px rgba(0,0,0,0.08); margin-bottom:10px;}
.metric-value{ font-size:28px; font-weight:700; color:#2563eb;}
.metric-label{ color:#64748b; font-size:14px; }
.stButton>button{ background:#2563eb; color:white; border-radius:10px; height:55px; font-size:18px; font-weight:bold; width:100%;}
</style>
""", unsafe_allow_html=True)

# ================== Load Files (Matching GitHub Structure) ==================
# بيتم تحديد المسار النسبي بناءً على مكان ملف الـ app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

DATA_PATH = os.path.join(BASE_DIR, "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model", "catboost_sales_model_v2.pkl")
FEAT_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")

@st.cache_resource
def load_essentials():
    # تحميل الداتا
    df_raw = pd.read_parquet(DATA_PATH)
    if 'InvoiceDate' in df_raw.columns:
        df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
        df_raw.set_index('InvoiceDate', inplace=True)
    
    # تحميل الموديل والميزات
    model_loaded = joblib.load(MODEL_PATH)
    features_loaded = joblib.load(FEAT_PATH)
    return df_raw, model_loaded, features_loaded

try:
    df, model, feature_names = load_essentials()
except Exception as e:
    st.error(f"⚠️ Error loading project files. Please ensure the 'data' and 'model' folders are present. Details: {e}")
    st.stop()

# ================== Header ==================
st.markdown(f"""
<div class='header-card'>
    <div class='name-title'>Eng. Goda Emad</div>
    <div class='project-title'>Smart Retail AI Platform</div>
    <div class='project-subtitle'>
        High-Precision Sales Forecasting using CatBoost | Model Accuracy (R2): 0.82
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Dashboard Controls ==================
c1, c2, c3, c4 = st.columns([2,2,2,1.5])

last_sales = float(df['Daily_Sales'].iloc[-1])
last_cust = int(df['Unique_Customers'].iloc[-1])

with c1:
    in_sales = st.number_input("Yesterday's Total Sales ($)", value=last_sales)
with c2:
    in_cust = st.number_input("Total Customers Yesterday", value=last_cust)
with c3:
    forecast_horizon = st.slider("Prediction Days", 7, 30, 14)
with c4:
    st.write("") # Spacer
    predict_btn = st.button("🔮 Run AI Model")

# ================== Forecast Engine ==================
def run_forecast(model_obj, df_history, feature_list, sales_input, cust_input, days_count):
    temp_df = df_history.copy()
    
    # تحديث آخر يوم بالمدخلات اليدوية
    temp_df.iloc[-1, temp_df.columns.get_loc('Daily_Sales')] = sales_input
    temp_df.iloc[-1, temp_df.columns.get_loc('Unique_Customers')] = cust_input
    
    forecast_results = []
    current_dt = temp_df.index.max()
    
    for i in range(days_count):
        next_dt = current_dt + timedelta(days=1)
        
        # تحضير الـ 21 ميزة المطلوبة للموديل
        row_data = {
            'Order_Count': temp_df['Order_Count'].mean(),
            'Unique_Customers': temp_df['Unique_Customers'].iloc[-1],
            'Total_Quantity': temp_df['Total_Quantity'].mean(),
            'Avg_Price': temp_df['Avg_Price'].mean(),
            'UK_Ratio': temp_df['UK_Ratio'].iloc[-1],
            'day': next_dt.day,
            'month': next_dt.month,
            'dayofweek': next_dt.dayofweek,
            'weekofyear': int(next_dt.isocalendar()[1]),
            'is_weekend': 1 if next_dt.dayofweek in [5, 6] else 0,
            'sales_lag_1': temp_df['Daily_Sales'].iloc[-1],
            'customers_lag_1': temp_df['Unique_Customers'].iloc[-1],
            'sales_lag_2': temp_df['Daily_Sales'].iloc[-2],
            'customers_lag_2': temp_df['Unique_Customers'].iloc[-2],
            'sales_lag_3': temp_df['Daily_Sales'].iloc[-3],
            'customers_lag_3': temp_df['Unique_Customers'].iloc[-3],
            'sales_lag_7': temp_df['Daily_Sales'].iloc[-7],
            'customers_lag_7': temp_df['Unique_Customers'].iloc[-7],
            'rolling_mean_7': temp_df['Daily_Sales'].tail(7).mean(),
            'rolling_std_7': temp_df['Daily_Sales'].tail(7).std()
        }
        
        # ترتيب المدخلات حسب الموديل
        X_vec = [row_data.get(f, 0) for f in feature_list]
        
        # التوقع
        prediction = model_obj.predict(X_vec)
        forecast_results.append(prediction)
        
        # تحديث التاريخ المرجعي للتوقع القادم
        new_entry = pd.DataFrame([row_data], index=[next_dt])
        new_entry['Daily_Sales'] = prediction
        temp_df = pd.concat([temp_df, new_entry])
        current_dt = next_dt
        
    return temp_df.tail(days_count)

# ================== Display Logic ==================
if predict_btn:
    with st.spinner('AI Engine is processing historical patterns...'):
        final_preds = run_forecast(model, df, feature_names, in_sales, in_cust, forecast_horizon)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-20:], y=df['Daily_Sales'].tail(20), name="Actual History", line=dict(color="#0f172a", width=2)))
        fig.add_trace(go.Scatter(x=final_preds.index, y=final_preds['Daily_Sales'], name="AI Forecast", line=dict(color="#2563eb", width=4, dash='dot')))
        
        fig.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Summary
        st.markdown("### 📊 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Tomorrow's Target</div><div class='metric-value'>${final_preds['Daily_Sales'].iloc[0]:,.0f}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Cumulative Forecast</div><div class='metric-value'>${final_preds['Daily_Sales'].sum():,.0f}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Peak Expected</div><div class='metric-value'>${final_preds['Daily_Sales'].max():,.0f}</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>Model Confidence</div><div class='metric-value'>82.09%</div></div>", unsafe_allow_html=True)

        # Download Report
        report_csv = final_preds[['Daily_Sales']].to_csv().encode('utf-8')
        st.download_button("📥 Download Forecast Report", report_csv, "retail_forecast_goda.csv", "text/csv")

st.markdown("<br><hr><center style='color:#64748b'>Retail Analytics AI Platform | Designed by Eng. Goda Emad © 2026</center>", unsafe_allow_html=True)
