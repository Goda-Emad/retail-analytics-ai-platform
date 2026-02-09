import streamlit as st
import pandas as pd
import joblib
import os

# إعداد الصفحة لتعمل بأقصى سرعة
st.set_page_config(page_title="Retail AI Forecast Dashboard", layout="wide")

st.title("📈 Retail Sales Forecasting AI Dashboard")

# تحديد المسارات بشكل ديناميكي لضمان عملها على السحابة (Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# نتحرك خطوة للخلف للوصول لمجلدات data و model كما في هيكل مشروعك
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

# ================== 1. تحميل البيانات (بصيغة Parquet البرق) ==================
@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# ================== 2. تحميل الموديل (مرة واحدة في الذاكرة) ==================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ خطأ في تحميل الملفات: {e}")
    st.info("تأكد من وجود ملف parquet في مجلد data وملف pkl في مجلد model")
    st.stop()

# ================== 3. عرض المبيعات التاريخية ==================
st.subheader("📊 Historical Daily Sales")

# نستخدم total_amount كما ظهر في الكود الأخير الخاص بك
daily = df[['InvoiceDate', 'total_amount']].sort_values('InvoiceDate')
daily = daily.set_index('InvoiceDate')

# رسم آخر 180 يوم فقط للسرعة (يمكنك تغيير الرقم أو حذفه)
st.line_chart(daily.tail(180))

# ================== 4. منطق التوقع (محسن بـ Cache) ==================
@st.cache_data
def generate_forecast(_model, _daily_data):
    last_date = _daily_data.index.max()
    future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]

    future_df = pd.DataFrame({'InvoiceDate': future_dates})
    future_df['day'] = future_df['InvoiceDate'].dt.day
    future_df['month'] = future_df['InvoiceDate'].dt.month
    future_df['year'] = future_df['InvoiceDate'].dt.year
    future_df['dayofweek'] = future_df['InvoiceDate'].dt.dayofweek

    # أخذ آخر القيم الحقيقية لبدء التوقع (استخدام آخر 30 قيمة لضمان توفر كل الـ Lags)
    last_values = list(_daily_data['total_amount'].tail(30))
    predictions = []

    for i in range(len(future_df)):
        # بناء صف الفيتشرز بناءً على التوقعات السابقة (Auto-regressive)
        # الترتيب: day, month, year, dayofweek, lag_1, lag_2, lag_3, lag_7
        l1, l2, l3, l7 = last_values[-1], last_values[-2], last_values[-3], last_values[-7]
        
        feat_cols = future_df.iloc[i][['day','month','year','dayofweek']].values
        features = list(feat_cols) + [l1, l2, l3, l7]
        
        # التوقع
        pred = _model.predict([features])[0]
        predictions.append(pred)
        last_values.append(pred)

    future_df['Predicted_Sales'] = predictions
    return future_df.set_index('InvoiceDate')

# تنفيذ التوقع مع رسالة انتظار احترافية
st.subheader("🔮 Forecast Next 30 Days")
with st.spinner('جاري حساب التوقعات باستخدام CatBoost...'):
    future_df = generate_forecast(model, daily)

# ================== 5. رسم المقارنة النهائية ==================
combined = pd.concat([
    daily.tail(60).rename(columns={'total_amount': 'Historical Sales'}),
    future_df.rename(columns={'Predicted_Sales': 'Forecasted Sales'})
], axis=1)

st.line_chart(combined)

st.success("✅ تم تحديث التوقعات بنجاح!")
