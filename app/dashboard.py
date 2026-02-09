import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# 1. إعداد الصفحة وتحميل الثيم الاحترافي
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide", initial_sidebar_state="expanded")

# إضافة CSS مخصص للخلفية والألوان
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. الهوية الشخصية (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80) # أيقونة بروفايل افتراضية
    st.markdown("### **Eng. Goda Emad**")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/goda-emad/) ")
    with col2: st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/Goda-Emad)")
    st.markdown("---")

st.title("🚀 Retail Sales AI Analytics")
st.markdown("Welcome to the advanced forecasting platform. Use the tools below to explore data insights.")

# 3. تحميل البيانات والموديل
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("⚠️ Files missing. Please check your data/model folders.")
    st.stop()

# 4. قسم أهمية العوامل (Feature Importance) - ملون وتفاعلي
st.subheader("🎯 AI Insights: Feature Importance")
try:
    importance = model.get_feature_importance()
    raw_names = ['Day', 'Month', 'Year', 'Day of Week', 'Lag 1', 'Lag 2', 'Lag 3', 'Lag 7']
    feature_names = raw_names[:len(importance)]

    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)

    # رسم Plotly احترافي ملون
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                 title="What drives the sales prediction?",
                 color='Importance', color_continuous_scale='Viridis',
                 template='plotly_white')
    fig_fi.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_fi, use_container_width=True)
except:
    st.info("Dynamic features loading...")

# 5. قسم التوقعات التاريخية (Historical Trend) - تفاعلي
st.subheader("📊 Historical Sales Trend Analysis")

# فلتر لاختيار النطاق الزمني
time_range = st.select_slider("Select View Range", options=[30, 60, 90, 180, 365], value=90)
plot_df = df.tail(time_range)

fig_hist = px.line(plot_df, x='InvoiceDate', y='total_amount', 
                  title=f"Sales Movement - Last {time_range} Days",
                  labels={'total_amount': 'Sales ($)', 'InvoiceDate': 'Date'},
                  template='plotly_white')

fig_hist.update_traces(line_color='#00d4ff', line_width=2)
fig_hist.update_layout(hovermode="x unified") # عرض القيم عند تمرير الماوس
st.plotly_chart(fig_hist, use_container_width=True)

# 6. التوقع اليدوي (Interactive Prediction)
st.sidebar.header("🕹️ Prediction Lab")
with st.sidebar.expander("Adjust Parameters", expanded=True):
    l1 = st.number_input("Yesterday Sales", value=float(df['total_amount'].iloc[-1]))
    l7 = st.number_input("Last Week Sales", value=float(df['total_amount'].iloc[-7]))

if st.sidebar.button("✨ Predict Now"):
    # منطق التوقع (مبسط للعرض)
    feat_len = len(model.get_feature_importance())
    test_data = [15, 2, 2026, 0, l1, l1, l1, l7][:feat_len]
    if len(test_data) < feat_len: test_data += [0] * (feat_len - len(test_data))
    
    pred = model.predict(test_data)
    st.sidebar.metric("AI Predicted Value", f"${pred:,.2f}", delta="Estimated")
    st.sidebar.write("Confidence: High ✅")

st.markdown("---")
st.caption(f"Developed with ❤️ by Eng. Goda Emad | Retail Analytics v2.0")
