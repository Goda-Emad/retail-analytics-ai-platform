import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import time
from utils import run_backtesting

# ================== 0️⃣ Model Version ==================
MODEL_VERSION = "v3.3"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# ================== 1️⃣ تحميل الأصول ==================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("catboost_sales_model_10features.pkl")
        scaler = joblib.load("scaler_10features.pkl")
        feature_names = joblib.load("feature_names_10features.pkl")
        df = pd.read_parquet("daily_sales_ready_10features.parquet")
        df.columns = [c.lower().strip() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
        return model, scaler, feature_names, df
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملفات: {e}")
        return None, None, None, None

model, scaler, feature_names, df_raw = load_assets()
if model is None:
    st.stop()

# ================== 2️⃣ معالجة Upload CSV ==================
def process_upload(file):
    uploaded_df = pd.read_csv(file)
    uploaded_df.columns = [c.lower().strip() for c in uploaded_df.columns]
    if 'date' in uploaded_df.columns:
        uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
        uploaded_df = uploaded_df.sort_values('date').set_index('date')
    return uploaded_df

# ================== 3️⃣ Sidebar ==================
st.sidebar.title(f"🚀 Control Center {MODEL_VERSION}")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()

# حماية Multi-Store
stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main"]
selected_store = st.sidebar.selectbox("🏪 Store", stores)
df_store = df_active[df_active['store_id'] == selected_store] if 'store_id' in df_active.columns else df_active

# Validation
if df_store.empty or len(df_store) < 30:
    st.error("⚠️ بيانات غير كافية أو الفرع غير موجود (نحتاج 30 يوم عمل).")
    st.stop()

horizon = st.sidebar.slider("مدة التوقع (أيام)", 7, 60, 14)
scenario = st.sidebar.select_slider("سيناريو السوق", options=["متشائم", "واقعي", "متفائل"], value="واقعي")
noise = st.sidebar.slider("مستوى التقلب", 0.0, 0.2, 0.05)
scenario_map = {"متشائم": 0.85, "واقعي": 1.0, "متفائل": 1.15}

# ================== 4️⃣ Backtesting ==================
@st.cache_data
def cached_backtesting(_df, _features, _scaler, _model):
    return run_backtesting(_df, _features, _scaler, _model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 5️⃣ Forecast Engine ==================
def generate_forecast(history_df, horizon, scenario_val, noise_val, residuals_std):
    np.random.seed(42)
    preds, lowers, uppers = [], [], []
    current_df = history_df[['sales']].copy()
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for i in range(horizon):
        next_date = current_df.index[-1] + timedelta(days=1)
        # بناء الميزات
        feat_dict = {
            'dayofweek_sin': np.sin(2*np.pi*next_date.dayofweek/7),
            'dayofweek_cos': np.cos(2*np.pi*next_date.dayofweek/7),
            'month_sin': np.sin(2*np.pi*(next_date.month-1)/12),
            'month_cos': np.cos(2*np.pi*(next_date.month-1)/12),
            'lag_1': float(current_df['sales'].iloc[-1]),
            'lag_7': float(current_df['sales'].iloc[-7] if len(current_df)>=7 else current_df['sales'].mean()),
            'rolling_mean_7': float(current_df['sales'].tail(7).mean()),
            'rolling_mean_14': float(current_df['sales'].tail(14).mean()),
            'is_weekend': 1 if next_date.dayofweek >= 5 else 0,
            'was_closed_yesterday': 1 if current_df['sales'].iloc[-1] == 0 else 0
        }
        # ترتيب الأعمدة بالضبط
        X_df = pd.DataFrame([feat_dict])[feature_names]
        try:
            X_df[num_cols] = scaler.transform(X_df[num_cols])
        except:
            X_df_scaled = scaler.transform(X_df)
            X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)
        
        # التوقع
        pred_log = model.predict(X_df)[0]
        pred_val = np.expm1(pred_log) * scenario_val
        pred_val *= (1 + np.random.normal(0, noise_val))
        pred_val = max(0, pred_val)
        
        # نطاق الثقة
        bound = 1.96 * residuals_std * np.sqrt(i + 1)
        preds.append(pred_val)
        lowers.append(max(0, pred_val - bound))
        uppers.append(pred_val + bound)
        
        # إضافة للتاريخ القادم
        new_row = pd.DataFrame({'sales': [pred_val]}, index=[next_date])
        current_df = pd.concat([current_df, new_row])
    
    return preds, lowers, uppers, current_df.index[-horizon:]

# ================== 6️⃣ Execution ==================
start_inf = time.time()
preds, lowers, uppers, forecast_dates = generate_forecast(
    df_store, horizon, scenario_map[scenario], noise, metrics['residuals_std']
)
inf_time = time.time() - start_inf

# ================== 7️⃣ Visualization & KPIs ==================
st.title(f"🚀 Retail AI Forecast | {selected_store}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Forecast", f"${np.sum(preds):,.0f}")
c2.metric("Model Accuracy (R²)", f"{metrics['r2']:.3f}")
c3.metric("Error Rate (MAPE)", f"{metrics['mape']*100:.2f}%")
c4.metric("Inference Time", f"{inf_time*1000:.1f} ms")

# الرسم البياني
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-45:], y=df_store['sales'].tail(45),
                         name="Actual Sales", line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=forecast_dates, y=preds,
                         name="AI Prediction", line=dict(color="#3b82f6", width=4)))
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
    y=np.concatenate([uppers, lowers[::-1]]),
    fill='toself', fillcolor='rgba(59,130,246,0.15)',
    line=dict(color='rgba(255,255,255,0)'), name="Confidence Band"
))
fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# الميزات وتحميل الملفات
st.subheader("🎯 Insights & Export")
col_a, col_b = st.columns(2)

with col_a:
    importance = model.get_feature_importance()
    fig_imp = go.Figure(go.Bar(x=importance, y=feature_names,
                               orientation='h', marker=dict(color='#3b82f6')))
    fig_imp.update_layout(template="plotly_dark", title="Feature Importance", height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

with col_b:
    res_df = pd.DataFrame({"Date": forecast_dates,
                           "Forecast": preds,
                           "Min_Expected": lowers,
                           "Max_Expected": uppers})
    st.write("📊 تقرير التوقعات القادم:")
    st.dataframe(res_df.head(), use_container_width=True)
    st.download_button("📥 Download Full CSV", res_df.to_csv(index=False),
                       f"forecast_{selected_store}.csv")

with st.expander("📝 System Diagnostics"):
    st.write(f"**Execution Log:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Backtesting Time:** {metrics['execution_time']:.2f}s")
    st.write(f"**Active Features:** {len(feature_names)}")
