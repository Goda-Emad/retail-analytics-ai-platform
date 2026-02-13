import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os
import time
from utils import run_backtesting

# ================== 0️⃣ Config & Background ==================
MODEL_VERSION = "v5.4 (Pro Custom)"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# إضافة الخلفية المناسبة للمشروع (Gradient Dark Blue)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b);
        color: white;
    }
    /* تحسين شكل الجداول */
    .stDataFrame {
        border: 1px solid #3b82f6;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ================== 1️⃣ Assets Loader (نفس كودك الأصلي) ==================
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
        st.error(f"❌ Error loading files: {e}")
        return None, None, None, None

model, scaler, feature_names, df_raw = load_assets()
if model is None: st.stop()

# ================== 2️⃣ Sidebar & Logic (نفس كودك الأصلي) ==================
lang_choice = st.sidebar.selectbox("🌐 Language / اللغة", options=["English", "عربي"])

def process_upload(file):
    uploaded_df = pd.read_csv(file)
    uploaded_df.columns = [c.lower().strip() for c in uploaded_df.columns]
    if 'date' in uploaded_df.columns:
        uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
        uploaded_df = uploaded_df.sort_values('date').set_index('date')
    return uploaded_df

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()

stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox("🏪 Select Store", stores)
df_store = df_active[df_active['store_id']==selected_store] if 'store_id' in df_active.columns else df_active

if len(df_store)<30:
    st.error("⚠️ Not enough data / بيانات غير كافية (30 يوم على الأقل)")
    st.stop()

forecast_horizon = st.sidebar.slider("📅 Forecast Days", 1, 60, 14)
scenario = st.sidebar.select_slider("Scenario", options=["متشائم","واقعي","متفائل"], value="واقعي")
scenario_map = {"متشائم":0.85,"واقعي":1.0,"متفائل":1.15}
noise = st.sidebar.slider("Market Volatility", 0.0, 0.2, 0.05)

@st.cache_data
def cached_backtesting(_df,_features,_scaler,_model):
    return run_backtesting(_df,_features,_scaler,_model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 3️⃣ Forecast Function (نفس كودك الأصلي) ==================
def generate_forecast(history_df, horizon, scenario_val, noise_val, residuals_std):
    np.random.seed(42)
    preds, lowers, uppers = [],[],[]
    current_df = history_df[['sales']].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    num_cols = ['lag_1','lag_7','rolling_mean_7','rolling_mean_14']
    MAX_SALES = max(100000, current_df['sales'].max()*3)
    for i in range(horizon):
        next_date = current_df.index[-1] + pd.Timedelta(days=1)
        feat_dict = {
            'dayofweek_sin': np.sin(2*np.pi*next_date.dayofweek/7),
            'dayofweek_cos': np.cos(2*np.pi*next_date.dayofweek/7),
            'month_sin': np.sin(2*np.pi*(next_date.month-1)/12),
            'month_cos': np.cos(2*np.pi*(next_date.month-1)/12),
            'lag_1': float(current_df['sales'].iloc[-1]),
            'lag_7': float(current_df['sales'].iloc[-7] if len(current_df)>=7 else current_df['sales'].mean()),
            'rolling_mean_7': float(current_df['sales'].tail(7).mean()),
            'rolling_mean_14': float(current_df['sales'].tail(14).mean()),
            'is_weekend': 1 if next_date.dayofweek>=5 else 0,
            'was_closed_yesterday': 1 if current_df['sales'].iloc[-1]<=0 else 0
        }
        X_df = pd.DataFrame([feat_dict])[feature_names].replace([np.inf,-np.inf],0)
        try: X_df[num_cols] = scaler.transform(X_df[num_cols])
        except: X_df = pd.DataFrame(scaler.transform(X_df), columns=feature_names, index=X_df.index)
        pred_log = model.predict(X_df)[0]
        pred_val = np.expm1(np.clip(pred_log,-10,15))*scenario_val
        pred_val *= (1+np.random.normal(0,noise_val))
        pred_val = np.clip(pred_val,0,MAX_SALES)
        bound = 1.96*residuals_std*np.sqrt(i+1)
        preds.append(pred_val)
        lowers.append(max(0,pred_val-bound))
        uppers.append(min(MAX_SALES*1.2,pred_val+bound))
        current_df.loc[next_date]=[pred_val]
    return preds, lowers, uppers, current_df.index[-horizon:]

preds, lowers, uppers, forecast_dates = generate_forecast(df_store, forecast_horizon, scenario_map[scenario], noise, metrics['residuals_std'])

# ================== 4️⃣ UI Updates (الألوان المطلوبة) ==================
st.title("📈 Retail Forecast | "+selected_store if lang_choice=="English" else "📈 التوقعات المستقبلية | "+selected_store)

m1,m2,m3,m4 = st.columns(4)
label1, label2, label3, label4 = ("Expected Total Sales", "R² Accuracy", "MAPE Error", "Inference Time") if lang_choice=="English" else ("إجمالي المبيعات", "دقة الموديل", "نسبة الخطأ", "زمن المعالجة")
m1.metric(label1, f"${np.sum(preds):,.0f}")
m2.metric(label2, f"{metrics['r2']:.3f}")
m3.metric(label3, f"{metrics['mape']*100:.2f}%")
m4.metric(label4, f"{time.time()-metrics['execution_time']:.2f} s")

# --- تنسيق الـ Chart بالألوان المطلوبة (Neon Blue & Glass effect) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-60:], y=df_store['sales'].tail(60), name="History", line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=forecast_dates, y=preds, name="AI Forecast", line=dict(color="#00f2fe", width=4)))
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
    y=np.concatenate([uppers, lowers[::-1]]),
    fill='toself', fillcolor='rgba(0, 242, 254, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="Confidence"
))
fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- تنسيق الجدول بالألوان (Color Gradient) ---
st.subheader("📥 Export & Data Preview" if lang_choice=="English" else "📥 معاينة البيانات والتصدير")
res_df = pd.DataFrame({"Date":forecast_dates,"Forecast":preds,"Min":lowers,"Max":uppers})

# إضافة ألوان داخل الجدول بناءً على القيم (الأعلى مبيعاً يأخذ لوناً مميزاً)
st.dataframe(res_df.style.background_gradient(cmap='Blues', subset=['Forecast']), use_container_width=True)

st.download_button("Download CSV" if lang_choice=="English" else "تحميل ملف CSV", res_df.to_csv(index=False), f"forecast_{selected_store}.csv")


# ================== Footer ==================
st.markdown("---")
footer_html = f"""
<div style="display:flex;justify-content:space-between;font-size:14px;color:#999;padding:10px 0">
<span>Eng.Goda Emad | <a href='https://github.com/Goda-Emad'>GitHub</a> | <a href='https://www.linkedin.com/in/goda-emad/'>LinkedIn</a></span>
<span>v{MODEL_VERSION}</span>
</div>
"""
st.markdown(footer_html,unsafe_allow_html=True)



