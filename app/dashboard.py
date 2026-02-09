import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. إعداد الصفحة
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# --- CSS الاحترافي: خلفية سوبر ماركت + تصميم زجاجي شفاف ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                    url('https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 3px solid #00d4ff;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-title {
        color: #0f172a;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل الملفات (المسارات)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model_v2.pkl")
FEAT_PATH = os.path.join(BASE_DIR, "..", "model", "feature_names.pkl")

@st.cache_data
def load_essentials():
    df = pd.read_parquet(DATA_PATH)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    return df, model, features

try:
    df, model, feature_names = load_essentials()
except Exception as e:
    st.error(f"⚠️ Error loading v2 files: {e}")
    st.stop()

# 3. الهوية الشخصية والـ Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Eng. Goda Emad</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00d4ff;'>AI & Retail Analytics Expert</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🛠️ LIVE Simulator")
    st.write("Adjust parameters to see AI react in real-time!")
    
    # محاكاة مبيعات "الأمس" و "الأسبوع الماضي"
    sim_lag1 = st.slider("Simulate Yesterday's Sales ($)", 0.0, float(df['total_amount'].max()), float(df['total_amount'].iloc[-1]))
    sim_lag7 = st.slider("Simulate Last Week's Sales ($)", 0.0, float(df['total_amount'].max()), float(df['total_amount'].iloc[-7]))
    
    st.markdown("---")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/goda-emad/) ")

# العنوان الرئيسي
st.markdown("<h1 class='main-title'>🛒 Smart Forecasting Platform v2</h1>", unsafe_allow_html=True)

# 4. دالة التوقع الذكي (The Pulse Logic)
def generate_dynamic_forecast(model, df, start_lag1, start_lag7, days=30):
    future_preds = []
    last_date = df['InvoiceDate'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # قائمة بآخر مبيعات لتحديث الـ Rolling Mean
    current_history = list(df['total_amount'].tail(30))
    current_history[-1] = start_lag1 # تحديث القيمة بناءً على الـ Slider
    
    for i in range(days):
        d = future_dates[i]
        # بناء الفيتشرز بنفس ترتيب الـ Colab بالضبط
        # features = ['day', 'month', 'dayofweek', 'is_weekend', 'rolling_mean_7', 'lag_1', 'lag_7']
        feat_values = {
            'day': d.day,
            'month': d.month,
            'dayofweek': d.dayofweek,
            'is_weekend': 1 if d.dayofweek in [4, 5] else 0,
            'rolling_mean_7': sum(current_history[-7:]) / 7,
            'lag_1': current_history[-1],
            'lag_7': current_history[-7] if len(current_history) >= 7 else current_history[-1]
        }
        
        # ترتيب القيم بناءً على الملف المحمل feature_names.pkl
        input_data = [feat_values[f] for f in feature_names]
        
        pred = model.predict(input_data)
        future_preds.append(pred)
        current_history.append(pred)
        
    return future_dates, future_preds

# 5. عرض النتائج التفاعلية
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Model Logic (Feature Importance)")
    importance = model.get_feature_importance()
    fi_df = pd.DataFrame({'Feature': feature_names, 'Value': importance}).sort_values('Value')
    fig_fi = px.bar(fi_df, x='Value', y='Feature', orientation='h', color='Value', 
                    color_continuous_scale='Viridis', template='plotly_white')
    fig_fi.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

with col2:
    st.subheader("📊 Sales History (Context)")
    fig_hist = px.line(df.tail(60), x='InvoiceDate', y='total_amount', template='plotly_white')
    fig_hist.update_traces(line_color='#0f172a', fill='tozeroy')
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# 6. قسم التوقع "النابض" التفاعلي
st.divider()
st.subheader("🔮 Interactive 30-Day AI Forecast")
st.info("The dotted blue line represents the AI's prediction. Move the sliders in the sidebar to see it change!")

f_dates, f_preds = generate_dynamic_forecast(model, df, sim_lag1, sim_lag7)
forecast_df = pd.DataFrame({'Date': f_dates, 'Sales': f_preds})

# رسم التوقع مع التاريخ
fig_final = go.Figure()
# التاريخ (الماضي)
fig_final.add_trace(go.Scatter(x=df['InvoiceDate'].tail(20), y=df['total_amount'].tail(20), 
                               name="Actual Sales", line=dict(color="#0f172a", width=3)))
# التوقع (المستقبل)
fig_final.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Sales'], 
                               name="AI Forecast", line=dict(color="#00d4ff", width=4, dash='dot')))

fig_final.update_layout(template='plotly_white', height=500, hovermode="x unified")
st.plotly_chart(fig_final, use_container_width=True)

st.markdown("---")
st.markdown(f"<center>Developed by <b>Eng. Goda Emad</b> | Powered by CatBoost v2 | {datetime.now().year}</center>", unsafe_allow_html=True)
