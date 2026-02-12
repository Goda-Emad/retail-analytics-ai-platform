import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import time
import os
from utils import run_backtesting

# ================== 0️⃣ Model Version & Config ==================
MODEL_VERSION = "v3.4"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# ================== 1️⃣ Smart Assets Loader ==================
@st.cache_resource
def load_assets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        paths = {
            "model": os.path.join(current_dir, "catboost_sales_model_10features.pkl"),
            "scaler": os.path.join(current_dir, "scaler_10features.pkl"),
            "features": os.path.join(current_dir, "feature_names_10features.pkl"),
            "data": os.path.join(current_dir, "daily_sales_ready_10features.parquet")
        }

        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        feature_names = joblib.load(paths["features"])
        df = pd.read_parquet(paths["data"])
        
        df.columns = [c.lower().strip() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            
        return model, scaler, feature_names, df
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملفات: {e}")
        return None, None, None, None

model, scaler, feature_names, df_raw = load_assets()
if model is None: st.stop()

# ================== 2️⃣ Logic Functions ==================
def process_upload(file):
    uploaded_df = pd.read_csv(file)
    uploaded_df.columns = [c.lower().strip() for c in uploaded_df.columns]
    if 'date' in uploaded_df.columns:
        uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
        uploaded_df = uploaded_df.sort_values('date').set_index('date')
    return uploaded_df

def generate_forecast(history_df, horizon, scenario_val, noise_val, residuals_std):
    np.random.seed(42)
    preds, lowers, uppers = [], [], []
    # تنظيف أولي لبيانات الإدخال
    current_df = history_df[['sales']].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for i in range(horizon):
        next_date = current_df.index[-1] + timedelta(days=1)
        
        # ميزات الأمان لضمان عدم وجود NaN
        try:
            lag_1 = float(current_df['sales'].iloc[-1])
            lag_7 = float(current_df['sales'].iloc[-7] if len(current_df)>=7 else current_df['sales'].mean())
            roll_7 = float(current_df['sales'].tail(7).mean())
            roll_14 = float(current_df['sales'].tail(14).mean())
        except:
            lag_1 = lag_7 = roll_7 = roll_14 = 0.0

        feat_dict = {
            'dayofweek_sin': np.sin(2*np.pi*next_date.dayofweek/7),
            'dayofweek_cos': np.cos(2*np.pi*next_date.dayofweek/7),
            'month_sin': np.sin(2*np.pi*(next_date.month-1)/12),
            'month_cos': np.cos(2*np.pi*(next_date.month-1)/12),
            'lag_1': lag_1,
            'lag_7': lag_7,
            'rolling_mean_7': roll_7,
            'rolling_mean_14': roll_14,
            'is_weekend': 1 if next_date.dayofweek >= 5 else 0,
            'was_closed_yesterday': 1 if lag_1 <= 0 else 0
        }
        
        X_df = pd.DataFrame([feat_dict])[feature_names]
        # 🛡️ الحماية من القيم غير المنتهية قبل الـ Scaler
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        try:
            X_df[num_cols] = scaler.transform(X_df[num_cols])
        except:
            X_df_scaled = scaler.transform(X_df)
            X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)
        
        # التوقع مع معالجة الـ Log
        pred_log = model.predict(X_df)[0]
        pred_val = np.expm1(pred_log) * scenario_val
        pred_val *= (1 + np.random.normal(0, noise_val))
        
        # منع القيم الشاذة (Outliers) الناتجة عن الـ expm1
        pred_val = np.nan_to_num(pred_val, nan=0.0, posinf=current_df['sales'].max()*1.5)
        pred_val = max(0, float(pred_val))
        
        bound = 1.96 * residuals_std * np.sqrt(i + 1)
        preds.append(pred_val)
        lowers.append(max(0, pred_val - bound))
        uppers.append(pred_val + bound)
        
        # تحديث السلسلة لليوم التالي
        new_row = pd.DataFrame({'sales': [pred_val]}, index=[next_date])
        current_df = pd.concat([current_df, new_row])
    
    return preds, lowers, uppers, current_df.index[-horizon:]

# ================== 3️⃣ Sidebar UI ==================
st.sidebar.title(f"🚀 Control Center {MODEL_VERSION}")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()

stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox("🏪 Select Store", stores)
df_store = df_active[df_active['store_id'] == selected_store] if 'store_id' in df_active.columns else df_active

if len(df_store) < 30:
    st.error("⚠️ بيانات غير كافية للتحليل (نحتاج 30 يوم عمل على الأقل).")
    st.stop()

st.sidebar.divider()
horizon = st.sidebar.slider("Forecast Days", 7, 60, 14)
scenario = st.sidebar.select_slider("Scenario", options=["متشائم", "واقعي", "متفائل"], value="واقعي")
scenario_map = {"متشائم": 0.85, "واقعي": 1.0, "متفائل": 1.15}
noise = st.sidebar.slider("Market Volatility", 0.0, 0.2, 0.05)

# ================== 4️⃣ Processing ==================
@st.cache_data
def cached_backtesting(_df, _features, _scaler, _model):
    return run_backtesting(_df, _features, _scaler, _model)

# تشغيل الـ Backtesting (تأكد أن utils.py محدث أيضاً)
metrics = cached_backtesting(df_store, feature_names, scaler, model)

# تشغيل التوقع المستقبلي
start_inf = time.time()
preds, lowers, uppers, forecast_dates = generate_forecast(
    df_store, horizon, scenario_map[scenario], noise, metrics['residuals_std']
)
inf_time = time.time() - start_inf

# ================== 5️⃣ Display ==================
st.title(f"🚀 Retail AI Forecast | {selected_store}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Expected Total Sales", f"${np.sum(preds):,.0f}")
m2.metric("Model R² Score", f"{metrics['r2']:.3f}")
m3.metric("Error Rate (MAPE)", f"{metrics['mape']*100:.2f}%")
m4.metric("Inference Time", f"{inf_time*1000:.1f} ms")

# الرسم البياني التفاعلي
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-45:], y=df_store['sales'].tail(45), name="Actual", line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=forecast_dates, y=preds, name="AI Forecast", line=dict(color="#3b82f6", width=4)))
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
    y=np.concatenate([uppers, lowers[::-1]]),
    fill='toself', fillcolor='rgba(59,130,246,0.15)', line=dict(color='rgba(255,255,255,0)'), name="95% Confidence"
))
fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# الجزء السفلي: الميزات والتحميل
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🎯 Feature Significance")
    importance = model.get_feature_importance()
    fig_imp = go.Figure(go.Bar(x=importance, y=feature_names, orientation='h', marker=dict(color='#3b82f6')))
    fig_imp.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

with col_b:
    st.subheader("📥 Export Results")
    res_df = pd.DataFrame({"Date": forecast_dates, "Forecast": preds, "Min_Bound": lowers, "Max_Bound": uppers})
    st.dataframe(res_df.head(10), use_container_width=True)
    st.download_button("Download Full CSV", res_df.to_csv(index=False), f"forecast_{selected_store}.csv")

with st.expander("📝 Diagnostics & System Logs"):
    st.write(f"**Version:** {MODEL_VERSION}")
    st.write(f"**Backtesting Records:** {metrics['data_points']}")
    st.write(f"**Features:** {', '.join(feature_names)}")
