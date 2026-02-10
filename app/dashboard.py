import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# ================== 1. Page Config & CSS ==================
st.set_page_config(page_title="Retail AI | Eng. Goda Emad", layout="wide")

st.markdown("""
<style>
.stApp { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #f0f4f8, #e0e7ff); }
.header-card { background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center; margin-bottom: 30px; }
.name-title { font-size: 38px; font-weight: 900; color: #0f172a; }
.project-title { font-size: 22px; font-weight: 700; color: #2563eb; }
.metric-card { background:white; padding:20px; border-radius:15px; text-align:center; box-shadow:0 5px 15px rgba(0,0,0,0.05); }
.metric-value{ font-size:26px; font-weight:700; color:#2563eb;}
.stButton>button{ background:#2563eb; color:white; border-radius:10px; height:50px; font-weight:bold; width:100%;}
</style>
""", unsafe_allow_html=True)

# ================== 2. Smart Path Finder (The Fix) ==================
def find_file(folder_name, file_name):
    # قائمة بالأماكن المحتملة للملفات بناءً على هيكل GitHub بتاعك
    base_paths = [
        os.getcwd(), # Root
        os.path.join(os.getcwd(), ".."), # Back one step
        os.path.dirname(os.path.abspath(__file__)), # Current script folder
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") # Script folder back one step
    ]
    
    for base in base_paths:
        full_path = os.path.join(base, folder_name, file_name)
        if os.path.exists(full_path):
            return full_path
    return None

# تحديد المسارات
DATA_PATH = find_file("data", "daily_sales_ready.parquet")
MODEL_PATH = find_file("model", "catboost_sales_model.pkl")
# تجربة المسار بالمسافة وبدونها لملف الميزات
FEAT_PATH = find_file("model", "feature_names .pkl") or find_file("model", "feature_names.pkl")

@st.cache_resource
def load_all():
    if not DATA_PATH or not MODEL_PATH or not FEAT_PATH:
        st.error("❌ File Loading Error!")
        st.info(f"Checking Path: {os.getcwd()}")
        st.stop()
        
    df = pd.read_parquet(DATA_PATH)
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.set_index('InvoiceDate', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    return df, model, features

df, model, feature_names = load_all()

# ================== 3. Header ==================
st.markdown(f"""
<div class='header-card'>
    <div class='name-title'>Eng. Goda Emad</div>
    <div class='project-title'>Smart Retail Sales Forecasting AI</div>
    <p style='color:#64748b;'>CatBoost Engine | Accuracy: 82% | Multi-Feature Prediction</p>
</div>
""", unsafe_allow_html=True)

# ================== 4. Inputs ==================
col1, col2, col3, col4 = st.columns([2,2,2,1.5])
with col1:
    in_sales = st.number_input("Yesterday's Sales ($)", value=float(df['Daily_Sales'].iloc[-1]))
with col2:
    in_cust = st.number_input("Yesterday's Customers", value=int(df['Unique_Customers'].iloc[-1]))
with col3:
    horizon = st.slider("Days to Forecast", 7, 30, 14)
with col4:
    st.write("")
    btn = st.button("🔮 Forecast")

# ================== 5. Forecast Logic ==================
def make_prediction(m, dh, fl, s_in, c_in, steps):
    temp = dh.copy()
    # تحديث آخر نقطة حقيقية بمدخلات المستخدم
    temp.iloc[-1, temp.columns.get_loc('Daily_Sales')] = s_in
    temp.iloc[-1, temp.columns.get_loc('Unique_Customers')] = c_in
    
    for _ in range(steps):
        nxt_dt = temp.index.max() + timedelta(days=1)
        # بناء ميزات اليوم الجديد
        row = {
            'Order_Count': temp['Order_Count'].mean(),
            'Unique_Customers': temp['Unique_Customers'].iloc[-1],
            'Total_Quantity': temp['Total_Quantity'].mean(),
            'Avg_Price': temp['Avg_Price'].mean(),
            'UK_Ratio': temp['UK_Ratio'].iloc[-1],
            'day': nxt_dt.day, 'month': nxt_dt.month, 'dayofweek': nxt_dt.dayofweek,
            'weekofyear': int(nxt_dt.isocalendar()[1]),
            'is_weekend': 1 if nxt_dt.dayofweek in [5, 6] else 0,
            'sales_lag_1': temp['Daily_Sales'].iloc[-1],
            'customers_lag_1': temp['Unique_Customers'].iloc[-1],
            'sales_lag_2': temp['Daily_Sales'].iloc[-2],
            'customers_lag_2': temp['Unique_Customers'].iloc[-2],
            'sales_lag_3': temp['Daily_Sales'].iloc[-3],
            'customers_lag_3': temp['Unique_Customers'].iloc[-3],
            'sales_lag_7': temp['Daily_Sales'].iloc[-7],
            'customers_lag_7': temp['Unique_Customers'].iloc[-7],
            'rolling_mean_7': temp['Daily_Sales'].tail(7).mean(),
            'rolling_std_7': temp['Daily_Sales'].tail(7).std()
        }
        # الترتيب الصحيح للموديل
        X = [row.get(f, 0) for f in fl]
        p = m.predict(X)
        
        # إضافة النتيجة للداتا لاستخدامها في اليوم اللي بعده
        new_df = pd.DataFrame([row], index=[nxt_dt])
        new_df['Daily_Sales'] = p
        temp = pd.concat([temp, new_df])
        
    return temp.tail(steps)

# ================== 6. Results & Visualization ==================
if btn:
    with st.spinner('AI is analyzing patterns...'):
        res = make_prediction(model, df, feature_names, in_sales, in_cust, horizon)
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-20:], y=df['Daily_Sales'].tail(20), name="History", line=dict(color="#0f172a")))
        fig.add_trace(go.Scatter(x=res.index, y=res['Daily_Sales'], name="Forecast", line=dict(color="#2563eb", width=4, dash='dot')))
        fig.update_layout(template="plotly_white", height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Tomorrow</div><div class='metric-value'>${res['Daily_Sales'].iloc[0]:,.0f}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Total Period</div><div class='metric-value'>${res['Daily_Sales'].sum():,.0f}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Model Confidence</div><div class='metric-value'>82.1%</div></div>", unsafe_allow_html=True)

st.markdown("<br><center style='color:#64748b'>Retail Analytics Dashboard | Eng. Goda Emad © 2026</center>", unsafe_allow_html=True)
