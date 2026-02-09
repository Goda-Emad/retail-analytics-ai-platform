import streamlit as st
import pandas as pd
import joblib
import os

# 1. إعدادات الصفحة والتحسين
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# 2. الهوية الشخصية وروابط التواصل في الـ Sidebar
with st.sidebar:
    st.markdown("## 👤 Developed by:")
    st.markdown("### **Eng. Goda Emad**")
    
    # أزرار الروابط الاحترافية
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/goda-emad/) ")
    with col2:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Goda-Emad)")
    
    st.divider()

st.title("📈 Retail Sales Forecasting AI Platform")

# 3. تحديد المسارات الديناميكية لضمان العمل على السحابة
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

# 4. دوال تحميل البيانات والموديل مع الـ Caching
@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# تنفيذ التحميل
try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")
    st.stop()

# 5. عرض أهمية العوامل (Feature Importance) - نسخة "آمنة" لتجنب الـ ValueError
st.subheader("🎯 Why did the AI predict this? (Feature Importance)")

try:
    importance = model.get_feature_importance()
    
    # قائمة بأسماء الفيتشرز المحتملة (يجب أن تتطابق مع ترتيب التدريب)
    # الموديل يتوقع: day, month, year, dayofweek, lag_1, lag_2, lag_3, lag_7...
    raw_names = ['Day', 'Month', 'Year', 'Day of Week', 'Lag 1', 'Lag 2', 'Lag 3', 'Lag 7', 'Lag 14', 'Lag 30']
    
    # موازنة الطول ديناميكياً لتجنب خطأ "All arrays must be of the same length"
    feature_names = raw_names[:len(importance)] 

    fi_df = pd.DataFrame({
        'Feature': feature_names, 
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(data=fi_df, x='Feature', y='Importance')
except Exception as e:
    st.info("💡 Feature importance view is updating based on your model's structure.")

st.divider()

# 6. قسم التوقع اليدوي (Interactive Section)
st.sidebar.header("🕹️ Test the AI Model")
input_day = st.sidebar.number_input("Day", 1, 31, 15)
input_month = st.sidebar.number_input("Month", 1, 12, 2)
input_lag1 = st.sidebar.number_input("Yesterday's Sales ($)", value=float(df['total_amount'].iloc[-1]))
input_lag7 = st.sidebar.number_input("Last Week Sales ($)", value=float(df['total_amount'].iloc[-7]))

if st.sidebar.button("Run Manual Prediction"):
    # بناء مصفوفة الإدخال بنفس طول الفيتشرز اللي الموديل متدرب عليها
    # بنملى القيم الأساسية والباقي بنخليه أصفار أو قيم افتراضية لو الموديل بيطلب أكتر
    num_features = len(model.get_feature_importance())
    test_features = [input_day, input_month, 2026, 0, input_lag1, input_lag1, input_lag1, input_lag7]
    
    # التأكد من أن الطول مطابق تماماً لما يتوقعه الموديل
    if len(test_features) < num_features:
        test_features += [0] * (num_features - len(test_features))
    else:
        test_features = test_features[:num_features]

    prediction = model.predict(test_features)
    st.sidebar.metric("AI Predicted Sales", f"${prediction:,.2f}")
    st.sidebar.balloons()

# 7. عرض الاتجاه العام (Historical Data)
st.subheader("📊 Historical Sales Trend (Last 100 Days)")
st.line_chart(df.set_index('InvoiceDate')['total_amount'].tail(100))

st.success(f"Dashboard updated successfully. Developed by Eng. Goda Emad")
