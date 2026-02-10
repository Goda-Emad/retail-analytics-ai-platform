import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI | Eng. Goda Emad", layout="wide")

st.markdown("""
<style>
.stApp { background: #f8fafc; }
.header-card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center; margin-bottom: 25px; border-top: 5px solid #2563eb; }
.metric-card { background:white; padding:20px; border-radius:12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
.metric-value{ font-size:24px; font-weight:700; color:#2563eb;}
</style>
""", unsafe_allow_html=True)

# ================== Direct Path Loading (Based on your Screenshot) ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# أسماء الملفات بالمسافات كما ظهرت في صورتك الأخيرة
DATA_FILE = "daily_sales_ready .parquet"  # لاحظ المسافة قبل النقطة
MODEL_FILE = "catboost_sales_model.pkl"
FEAT_FILE = "feature_names .pkl"         # لاحظ المسافة قبل النقطة

@st.cache_resource
def load_data():
    d_path = os.path.join(CURRENT_DIR, DATA_FILE)
    m_path = os.path.join(CURRENT_DIR, MODEL_FILE)
    f_path = os.path.join(CURRENT_DIR, FEAT_FILE)
    
    # التأكد من وجود الملفات
    if not os.path.exists(d_path):
        st.error(f"❌ File not found: {DATA_FILE}")
        st.stop()
        
    return pd.read_parquet(d_path), joblib.load(m_path), joblib.load(f_path)

try:
    df, model, feature_names = load_data()
    df.index = pd.to_datetime(df.index)
except Exception as e:
    st.error(f"Error Loading Files: {e}")
    st.stop()

# ================== UI & Header ==================
st.markdown(f"""
<div class='header-card'>
    <h1 style='margin:0; color:#0f172a;'>Eng. Goda Emad</h1>
    <h3 style='margin:0; color:#2563eb;'>Smart Retail AI Platform</h3>
    <p style='color:#64748b; margin-top:5px;'>High-Precision Forecasting (R2: 82.09%)</p>
</div>
""", unsafe_allow_html=True)

# ================== Sidebar Inputs ==================
with st.sidebar:
    st.header("📊 Parameters")
    last_val = float(df['Daily_Sales'].iloc[-1])
    in_sales = st.number_input("Last Day Sales ($)", value=last_val)
    in_cust = st.number_input("Last Day Customers", value=int(df['Unique_Customers'].iloc[-1]))
    horizon = st.slider("Forecast Days", 7, 30, 14)
    predict_btn = st.button("🔮 Run Forecast", use_container_width=True)

# ================== Forecast Logic ==================
if predict_btn:
    temp = df.copy()
    temp.iloc[-1, temp.columns.get_loc('Daily_Sales')] = in_sales
    temp.iloc[-1, temp.columns.get_loc('Unique_Customers')] = in_cust
    
    for _ in range(horizon):
        next_dt = temp.index.max() + timedelta(days=1)
        row = {
            'Order_Count': temp['Order_Count'].mean(),
            'Unique_Customers': temp['Unique_Customers'].iloc[-1],
            'Total_Quantity': temp['Total_Quantity'].mean(),
            'Avg_Price': temp['Avg_Price'].mean(),
            'UK_Ratio': temp['UK_Ratio'].iloc[-1],
            'day': next_dt.day, 'month': next_dt.month, 'dayofweek': next_dt.dayofweek,
            'weekofyear': int(next_dt.isocalendar()[1]),
            'is_weekend': 1 if next_dt.dayofweek in [5, 6] else 0,
            'sales_lag_1': temp['Daily_Sales'].iloc[-1], 'customers_lag_1': temp['Unique_Customers'].iloc[-1],
            'sales_lag_2': temp['Daily_Sales'].iloc[-2], 'customers_lag_2': temp['Unique_Customers'].iloc[-2],
            'sales_lag_3': temp['Daily_Sales'].iloc[-3], 'customers_lag_3': temp['Unique_Customers'].iloc[-3],
            'sales_lag_7': temp['Daily_Sales'].iloc[-7], 'customers_lag_7': temp['Unique_Customers'].iloc[-7],
            'rolling_mean_7': temp['Daily_Sales'].tail(7).mean(),
            'rolling_std_7': temp['Daily_Sales'].tail(7).std()
        }
        X = [row.get(f, 0) for f in feature_names]
        pred = model.predict(X)
        new_entry = pd.DataFrame([row], index=[next_dt])
        new_entry['Daily_Sales'] = pred
        temp = pd.concat([temp, new_entry])
    
    res = temp.tail(horizon)

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-20:], y=df['Daily_Sales'].tail(20), name="History", line=dict(color="#0f172a")))
    fig.add_trace(go.Scatter(x=res.index, y=res['Daily_Sales'], name="AI Forecast", line=dict(color="#2563eb", width=4)))
    fig.update_layout(template="plotly_white", height=450, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-card'><div style='color:#64748b'>Next Day</div><div class='metric-value'>${res['Daily_Sales'].iloc[0]:,.0f}</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div style='color:#64748b'>Total Period</div><div class='metric-value'>${res['Daily_Sales'].sum():,.0f}</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div style='color:#64748b'>Confidence</div><div class='metric-value'>82%</div></div>", unsafe_allow_html=True)

else:
    st.info("👈 Use the sidebar to set parameters and click Forecast.")

st.markdown("<br><hr><center style='color:#64748b'>Retail Analytics AI | Eng. Goda Emad © 2026</center>", unsafe_allow_html=True)
