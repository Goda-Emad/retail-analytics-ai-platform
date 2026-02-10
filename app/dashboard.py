import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# ================== Page Config ==================
st.set_page_config(page_title="Retail AI Forecasting | Eng. Goda Emad", layout="wide")

# ================== Premium CSS (Your Professional Style) ==================
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
.metric-card { background:white; padding:20px; border-radius:18px; text-align:center; box-shadow:0 8px 20px rgba(0,0,0,0.08); margin-bottom:10px;}
.metric-value{ font-size:28px; font-weight:700; color:#2563eb;}
.stButton>button{ background:#2563eb; color:white; border-radius:10px; height:55px; font-size:18px; font-weight:bold; width:100%;}
</style>
""", unsafe_allow_html=True)

# ================== Fixed Path Logic (Based on your Screenshots) ==================
# بما أن ملف dashboard.py جوه فولدر app، لازم نرجع خطوة لورا بالـ (..) لنوصل للـ data والـ model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# حل مشكلة المسافات والأسماء من الصور
DATA_PATH = os.path.join(BASE_DIR, "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model", "catboost_sales_model.pkl")

# هنا اللعبة: الكود هيدور على الملف بالمسافة وبدونها عشان ميعملش Error
FEAT_PATH_1 = os.path.join(BASE_DIR, "model", "feature_names .pkl") # بالمسافة كما في الصورة
FEAT_PATH_2 = os.path.join(BASE_DIR, "model", "feature_names.pkl")  # بدون مسافة احتياطي

@st.cache_resource
def load_essentials():
    # 1. تحميل الداتا
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data missing at: {DATA_PATH}")
    df_raw = pd.read_parquet(DATA_PATH)
    if 'InvoiceDate' in df_raw.columns:
        df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
        df_raw.set_index('InvoiceDate', inplace=True)
    
    # 2. تحميل الموديل
    model_loaded = joblib.load(MODEL_PATH)
    
    # 3. تحميل الميزات (محاولة بكلتا الطريقتين)
    final_feat_path = FEAT_PATH_1 if os.path.exists(FEAT_PATH_1) else FEAT_PATH_2
    features_loaded = joblib.load(final_feat_path)
    
    return df_raw, model_loaded, features_loaded

try:
    df, model, feature_names = load_essentials()
except Exception as e:
    st.error(f"⚠️ Error: {e}")
    st.info(f"Looking in: {BASE_DIR}")
    st.stop()

# ================== Header ==================
st.markdown(f"""
<div class='header-card'>
    <div class='name-title'>Eng. Goda Emad</div>
    <div class='project-title'>Smart Retail Sales Forecasting</div>
    <div style='color:#64748b;'>Model Accuracy: 82% | 21 Memory Features Enabled</div>
</div>
""", unsafe_allow_html=True)

# ================== Controls ==================
c1, c2, c3, c4 = st.columns([2,2,2,1.5])
with c1: in_sales = st.number_input("Last Sales ($)", value=float(df['Daily_Sales'].iloc[-1]))
with c2: in_cust = st.number_input("Last Customers", value=int(df['Unique_Customers'].iloc[-1]))
with c3: forecast_h = st.slider("Days", 7, 30, 14)
with c4: 
    st.write("") 
    predict_btn = st.button("🔮 Forecast")

# ================== Engine ==================
def run_forecast(m, dh, fl, s_in, c_in, horizon):
    tmp = dh.copy()
    tmp.iloc[-1, tmp.columns.get_loc('Daily_Sales')] = s_in
    tmp.iloc[-1, tmp.columns.get_loc('Unique_Customers')] = c_in
    
    for _ in range(horizon):
        next_dt = tmp.index.max() + timedelta(days=1)
        row = {
            'Order_Count': tmp['Order_Count'].mean(),
            'Unique_Customers': tmp['Unique_Customers'].iloc[-1],
            'Total_Quantity': tmp['Total_Quantity'].mean(),
            'Avg_Price': tmp['Avg_Price'].mean(),
            'UK_Ratio': tmp['UK_Ratio'].iloc[-1],
            'day': next_dt.day, 'month': next_dt.month, 'dayofweek': next_dt.dayofweek,
            'weekofyear': int(next_dt.isocalendar()[1]),
            'is_weekend': 1 if next_dt.dayofweek in [5, 6] else 0,
            'sales_lag_1': tmp['Daily_Sales'].iloc[-1], 'customers_lag_1': tmp['Unique_Customers'].iloc[-1],
            'sales_lag_2': tmp['Daily_Sales'].iloc[-2], 'customers_lag_2': tmp['Unique_Customers'].iloc[-2],
            'sales_lag_3': tmp['Daily_Sales'].iloc[-3], 'customers_lag_3': tmp['Unique_Customers'].iloc[-3],
            'sales_lag_7': tmp['Daily_Sales'].iloc[-7], 'customers_lag_7': tmp['Unique_Customers'].iloc[-7],
            'rolling_mean_7': tmp['Daily_Sales'].tail(7).mean(),
            'rolling_std_7': tmp['Daily_Sales'].tail(7).std()
        }
        # ترتيب المدخلات حسب الموديل
        X_vec = [row.get(f, 0) for f in fl]
        pred = m.predict(X_vec)
        
        # إضافة السطر الجديد
        new_row = pd.DataFrame([row], index=[next_dt])
        new_row['Daily_Sales'] = pred
        tmp = pd.concat([tmp, new_row])
    return tmp.tail(horizon)

# ================== Plot ==================
if predict_btn:
    results = run_forecast(model, df, feature_names, in_sales, in_cust, forecast_h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-20:], y=df['Daily_Sales'].tail(20), name="History", line=dict(color="#0f172a")))
    fig.add_trace(go.Scatter(x=results.index, y=results['Daily_Sales'], name="Forecast", line=dict(color="#2563eb", width=4, dash='dot')))
    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-card'><div class='metric-value'>${results['Daily_Sales'].iloc[0]:,.0f}</div><div style='color:#64748b'>Tomorrow</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-value'>${results['Daily_Sales'].sum():,.0f}</div><div style='color:#64748b'>Total Period</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-value'>82%</div><div style='color:#64748b'>Confidence</div></div>", unsafe_allow_html=True)

st.markdown("<br><center style='color:#64748b'>© 2026 Eng. Goda Emad</center>", unsafe_allow_html=True)
