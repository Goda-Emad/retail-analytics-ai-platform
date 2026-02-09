import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. إعداد الصفحة
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# --- CSS الاحترافي: خلفية سوبر ماركت + زجاجية (Glassmorphism) ---
st.markdown("""
    <style>
    /* خلفية السوبر ماركت مع تمويه خفيف لزيادة التركيز على البيانات */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                    url('https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
    }

    /* تنسيق العمود الجانبي (Sidebar) ليكون كحلي غامق واحترافي */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.98) !important;
        border-right: 3px solid #00d4ff;
    }
    
    /* جعل كل نصوص الـ Sidebar بيضاء وواضحة جداً */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }

    /* تنسيق الكروت البيضاء الزجاجية للرسومات */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,212,255,0.2);
    }

    /* تحسين شكل الأزرار والمداخل */
    .stButton > button {
        width: 100%;
        background-color: #00d4ff !important;
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    
    /* عنوان الصفحة الرئيسي */
    .main-title {
        color: #0f172a;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: -50px;
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. الهوية الشخصية (Sidebar)
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("<h2 style='margin-bottom: 0;'>Eng. Goda Emad</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #00d4ff !important;'>AI & Data Science Engineer</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    # روابط التواصل بشكل أزرار Badges
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/goda-emad/) ")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View_Code-lightgrey?style=for-the-badge&logo=github)](https://github.com/Goda-Emad)")
    st.markdown("---")
    
    # معمل التوقعات (Inputs)
    st.markdown("### 🕹️ Prediction Lab")
    in_lag1 = st.number_input("Sales Yesterday ($)", value=0.0)
    in_lag7 = st.number_input("Sales Last Week ($)", value=0.0)
    in_day = st.slider("Target Day of Month", 1, 31, 15)
    
    run_pred = st.button("✨ Run AI Analysis")

# 3. تحميل الملفات (المسارات)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

@st.cache_data
def load_data(path):
    data = pd.read_parquet(path)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    return data

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# تنفيذ التحميل مع رسالة تنبيه في حالة الخطأ
try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
    # تحديث القيم الافتراضية للـ Inputs بناءً على آخر بيانات
    if in_lag1 == 0.0: in_lag1 = float(df['total_amount'].iloc[-1])
    if in_lag7 == 0.0: in_lag7 = float(df['total_amount'].iloc[-7])
except Exception as e:
    st.error(f"⚠️ Error: Make sure model and data files are in place.")
    st.stop()

# العنوان الرئيسي في الصفحة
st.markdown("<h1 class='main-title'>🛒 Smart Retail Forecasting AI</h1>", unsafe_allow_html=True)

# 4. الرسوم البيانية التفاعلية (Layout)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🎯 Why did AI predict this?")
    # حساب أهمية العوامل ديناميكياً
    importance = model.get_feature_importance()
    raw_names = ['Day', 'Month', 'Year', 'WeekDay', 'Lag 1', 'Lag 2', 'Lag 3', 'Lag 7', 'Rolling Mean']
    feat_names = raw_names[:len(importance)]
    
    fi_df = pd.DataFrame({'Feature': feat_names, 'Power': importance}).sort_values('Power')
    
    fig_fi = px.bar(fi_df, x='Power', y='Feature', orientation='h',
                    color='Power', color_continuous_scale='Bluered',
                    template='plotly_white')
    fig_fi.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

with col_right:
    st.markdown("### 📊 Historical Sales Trend")
    # رسم بياني للمبيعات التاريخية (آخر 90 يوم)
    hist_plot_df = df.tail(90)
    fig_hist = px.area(hist_plot_df, x='InvoiceDate', y='total_amount',
                       labels={'total_amount': 'Sales ($)', 'InvoiceDate': 'Date'},
                       template='plotly_white')
    fig_hist.update_traces(line_color='#00d4ff')
    fig_hist.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

# 5. قسم التوقع المستقبلي (30 يوم)
st.divider()
st.markdown("### 🔮 AI Future Sales Forecast (Next 30 Days)")

# حساب التوقعات (Auto-regression Loop)
last_date = df['InvoiceDate'].max()
future_dates = pd.date_range(start=last_date, periods=31)[1:]

try:
    preds = []
    current_history = list(df['total_amount'].tail(30))
    
    for i in range(30):
        # بناء مصفوفة الميزات (تأكد أنها مطابقة للموديل)
        curr_date = future_dates[i]
        features = [curr_date.day, curr_date.month, 2026, curr_date.dayofweek, 
                    current_history[-1], current_history[-2], current_history[-3], current_history[-7]]
        # قص أو زيادة الفيتشرز لتناسب الموديل
        features = features[:len(importance)]
        
        p = model.predict(features)
        preds.append(p)
        current_history.append(p)

    f_df = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
    
    # رسم يجمع الماضي والمستقبل
    fig_final = go.Figure()
    fig_final.add_trace(go.Scatter(x=df['InvoiceDate'].tail(40), y=df['total_amount'].tail(40), 
                                   name="Past Sales", line=dict(color="#0f172a", width=2)))
    fig_final.add_trace(go.Scatter(x=f_df['Date'], y=f_df['Forecast'], 
                                   name="AI Forecast", line=dict(color="#00d4ff", width=4, dash='dot')))
    
    fig_final.update_layout(template='plotly_white', height=450, hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_final, use_container_width=True)
except Exception as e:
    st.warning("AI is calculating the next steps...")

# زر التوقع اليدوي (Action)
if run_pred:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Quick Result")
    # توقع بسيط سريع للقيم المدخلة
    manual_feat = [in_day, datetime.now().month, 2026, 0, in_lag1, in_lag1, in_lag1, in_lag7][:len(importance)]
    res = model.predict(manual_feat)
    st.sidebar.metric("Predicted Amount", f"${res:,.2f}")
    st.sidebar.balloons()

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #64748b;'>Developed with ❤️ by Eng. Goda Emad | Retail Analytics Platform © 2026</p>", unsafe_allow_html=True)
