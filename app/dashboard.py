import streamlit as st
import pandas as pd
import joblib
import os

# إعداد الصفحة
st.set_page_config(page_title="Retail AI Pro Dashboard", layout="wide")

# ================== Branding & Social Links (Sidebar) ==================
with st.sidebar:
    st.markdown(f"## 👤 Developed by:")
    st.markdown(f"### **Eng.Goda Emad**")
    
    # روابط التواصل الاجتماعي بشكل أزرار أنيقة
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/goda-emad/) ")
    with col2:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Goda-Emad)")
    
    st.divider()

st.title("📈 Retail Sales AI: Features & Forecasting")

# تحديد المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# تحميل البيانات والموديل
try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- عرض أهمية العوامل (Feature Importance) ---
st.subheader("🎯 Why did the AI predict this? (Feature Importance)")
importance = model.get_feature_importance()
feature_names = ['day', 'month', 'year', 'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'lag_7']
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
st.bar_chart(data=fi_df, x='Feature', y='Importance')

st.divider()

# --- التوقع اليدوي (Interactive Section) ---
st.sidebar.header("🕹️ Test the AI Model")
input_day = st.sidebar.number_input("Day", 1, 31, 10)
input_month = st.sidebar.number_input("Month", 1, 12, 5)
input_lag1 = st.sidebar.number_input("Yesterday's Sales ($)", value=float(df['total_amount'].iloc[-1]))
input_lag7 = st.sidebar.number_input("Last Week Sales ($)", value=float(df['total_amount'].iloc[-7]))

if st.sidebar.button("Run Manual Prediction"):
    test_features = [input_day, input_month, 2026, 1, input_lag1, input_lag1, input_lag1, input_lag7]
    prediction = model.predict(test_features)
    st.sidebar.metric("AI Prediction", f"${prediction:,.2f}")
    st.sidebar.balloons()

# --- عرض البيانات التاريخية والتوقعات العامة ---
st.subheader("📊 Historical Sales & AI Forecast")
st.line_chart(df.set_index('InvoiceDate')['total_amount'].tail(100))

st.success(f"Dashboard is live! Great job, Eng. Goda.")
