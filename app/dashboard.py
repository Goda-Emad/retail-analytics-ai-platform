import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import time
import os
from utils import run_backtesting 

# ================== 1. إعدادات الصفحة والجماليات ==================
st.set_page_config(page_title="Retail AI Pro Max", layout="wide", page_icon="📈")

# CSS لإضافة لمسة احترافية
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# ================== 2. تحميل الأصول (مع Cache) ==================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("catboost_sales_model_10features.pkl")
        scaler = joblib.load("scaler_10features.pkl")
        features = joblib.load("feature_names_10features.pkl")
        df = pd.read_parquet("daily_sales_ready_10features.parquet")
        # تنظيف البيانات
        df.columns = [str(c).lower().strip() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
        return model, scaler, features, df
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملفات الأساسية: {e}")
        return None, None, None, None

model, scaler, feature_names, df_init = load_assets()

# ================== 3. وظيفة التوقع (المحرك) ==================
def generate_forecast(history_df, horizon, scenario_factor, noise_val, residuals_std):
    start_time = time.time()
    preds, lowers, uppers = [], [], []
    current_df = history_df[['sales']].copy()
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for i in range(horizon):
        next_date = current_df.index[-1] + timedelta(days=1)
        feat_dict = {
            'dayofweek_sin': np.sin(2 * np.pi * next_date.dayofweek / 7),
            'dayofweek_cos': np.cos(2 * np.pi * next_date.dayofweek / 7),
            'month_sin': np.sin(2 * np.pi * (next_date.month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (next_date.month - 1) / 12),
            'lag_1': float(current_df['sales'].iloc[-1]),
            'lag_7': float(current_df['sales'].iloc[-7] if len(current_df)>=7 else current_df['sales'].mean()),
            'rolling_mean_7': float(current_df['sales'].tail(7).mean()),
            'rolling_mean_14': float(current_df['sales'].tail(14).mean()),
            'is_weekend': 1 if next_date.dayofweek >= 5 else 0,
            'was_closed_yesterday': 1 if current_df['sales'].iloc[-1] == 0 else 0
        }
        
        X_df = pd.DataFrame([feat_dict])[feature_names] # ترتيب الأعمدة
        X_df[num_cols] = scaler.transform(X_df[num_cols]) # الـ Scaling
        
        pred = np.expm1(model.predict(X_df)[0]) * scenario_factor
        pred_final = max(0, pred * (1 + np.random.normal(0, noise_val)))
        
        # Confidence Interval حقيقي
        bound = (i + 1)**0.5 * residuals_std 
        
        preds.append(pred_final)
        lowers.append(max(0, pred_final - bound))
        uppers.append(pred_final + bound)
        current_df = pd.concat([current_df, pd.Series([pred_final], index=[next_date], name='sales').to_frame()])
    
    return preds, lowers, uppers, current_df.index[-horizon:], time.time() - start_time

# ================== 4. واجهة المستخدم (Sidebar) ==================
if model is not None:
    st.sidebar.title("🎮 لوحة التحكم")
    
    # ميزة 7: رفع ملف جديد
    uploaded_file = st.sidebar.file_uploader("📂 ارفع بيانات جديدة (CSV)", type="csv")
    if uploaded_file:
        df_init = pd.read_csv(uploaded_file, index_col='date', parse_dates=True)
        st.sidebar.success("تم استخدام البيانات المرفوعة!")

    # ميزة 8: دعم فروع متعددة (Multi-Store)
    stores = df_init['store_id'].unique() if 'store_id' in df_init.columns else ["الفرع الرئيسي"]
    selected_store = st.sidebar.selectbox("🏪 اختر الفرع", stores)
    
    # فلترة البيانات بناءً على الفرع
    if 'store_id' in df_init.columns:
        df_final = df_init[df_init['store_id'] == selected_store]
    else:
        df_final = df_init

    st.sidebar.divider()
    horizon = st.sidebar.slider("مدة التوقع (أيام)", 7, 60, 14)
    scenario = st.sidebar.select_slider("سيناريو السوق", options=["متشائم", "واقعي", "متفائل"], value="واقعي")
    sc_map = {"متشائم": 0.85, "واقعي": 1.0, "متفائل": 1.15}
    noise = st.sidebar.slider("مستوى التقلب", 0.0, 0.2, 0.05)

    # ================== 5. العرض الرئيسي والـ Metrics ==================
    st.title("🚀 Retail AI Forecast Engine")
    st.markdown(f"عرض البيانات لـ: **{selected_store}**")

    # حساب الـ Backtesting (Cached)
    with st.spinner("🔍 جاري تحليل دقة الموديل..."):
        metrics = run_backtesting(df_final, feature_names, scaler, model)

    # تنفيذ التوقع
    preds, lowers, uppers, dates, inf_time = generate_forecast(
        df_final, horizon, sc_map[scenario], noise, metrics['residuals_std']
    )

    # ميزة 1 & 5: عرض KPIs حقيقية وزمن التنفيذ
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("التوقع الكلي", f"${np.sum(preds):,.0f}")
    c2.metric("دقة الموديل (R²)", f"{metrics['r2']*100:.1f}%")
    c3.metric("نسبة الخطأ (MAPE)", f"{metrics['mape']*100:.2f}%")
    c4.metric("زمن المعالجة", f"{inf_time*1000:.1f} ms")

    # ================== 6. الرسم البياني المتقدم ==================
    fig = go.Figure()
    # بيانات تاريخية
    fig.add_trace(go.Scatter(x=df_final.index[-45:], y=df_final['sales'].tail(45), name="مبيعات حقيقية", line=dict(color="#94a3b8")))
    # نطاق الثقة (Confidence Interval)
    fig.add_trace(go.Scatter(x=np.concatenate([dates, dates[::-1]]), y=np.concatenate([uppers, lowers[::-1]]),
                             fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="نطاق الشك"))
    # التوقع
    fig.add_trace(go.Scatter(x=dates, y=preds, name="توقع الذكاء الاصطناعي", line=dict(color="#3b82f6", width=4)))

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ميزة 6: تحميل التوقعات
    st.divider()
    res_df = pd.DataFrame({"Date": dates, "Forecast": preds, "Upper": uppers, "Lower": lowers})
    st.download_button(label="📥 تحميل تقرير التوقعات (CSV)", data=res_df.to_csv().encode('utf-8'),
                       file_name=f'forecast_{selected_store}.csv', mime='text/csv')

    # ميزة 9: Logging بسيط في الصفحة
    with st.expander("🛠️ تفاصيل فنية (System Logs)"):
        st.write(f"عدد السجلات المستخدمة: {len(df_final)}")
        st.write(f"الميزات المستخدمة: {', '.join(feature_names)}")
        st.write(f"حالة الموديل: Stable")
