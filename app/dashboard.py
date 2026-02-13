import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os
import time
from utils import run_backtesting

# ================== Config ==================
MODEL_VERSION = "v5.3 (Pro Interface)"
st.set_page_config(page_title=f"📈 Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# ================== Load Assets ==================
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(current_dir, "catboost_sales_model_10features.pkl"))
        scaler = joblib.load(os.path.join(current_dir, "scaler_10features.pkl"))
        feature_names = joblib.load(os.path.join(current_dir, "feature_names_10features.pkl"))
        df = pd.read_parquet(os.path.join(current_dir, "daily_sales_ready_10features.parquet"))
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

# ================== Sidebar ==================
st.sidebar.title("🚀 Retail AI Control Center")
theme_choice = st.sidebar.selectbox("🖌️ Theme", ["Light","Dark"])
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")

if uploaded_file:
    df_active = pd.read_csv(uploaded_file)
    df_active.columns = [c.lower().strip() for c in df_active.columns]
    if 'date' in df_active.columns:
        df_active['date'] = pd.to_datetime(df_active['date'])
        df_active = df_active.sort_values('date').set_index('date')
else:
    df_active = df_raw.copy()

stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox("🏪 Select Store", stores)
df_store = df_active[df_active['store_id']==selected_store] if 'store_id' in df_active.columns else df_active

horizons = st.sidebar.multiselect("📆 Forecast Periods (Days)", options=[7,14,30], default=[7,14,30])
noise = st.sidebar.slider("Market Volatility", 0.0, 0.2, 0.05)

# ================== Theme ==================
if theme_choice == "Light":
    bg_color, text_color, chart_template = "#FFFFFF", "#000000", "plotly_white"
else:
    bg_color, text_color, chart_template = "#0E1117", "#F5F5F5", "plotly_dark"

st.markdown(f"""
    <style>
        .main {{background-color: {bg_color}; color: {text_color};}}
        .stMetric {{background-color: {'#eef2f7' if theme_choice=='Light' else '#1e293b'}; border-radius:8px; padding:10px;}}
    </style>
""", unsafe_allow_html=True)

# ================== Backtesting ==================
@st.cache_data
def cached_backtesting(_df, _features, _scaler, _model):
    return run_backtesting(_df, _features, _scaler, _model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== Forecast Function ==================
def generate_forecast(df, horizon_list, scenario_val=1.0, noise_val=0.05):
    np.random.seed(42)
    forecasts = {}
    for horizon in horizon_list:
        preds, lowers, uppers = [], [], []
        current_df = df[['sales']].copy().replace([np.inf,-np.inf],0).fillna(0)
        MAX_SALES = max(100000, current_df['sales'].max()*3)
        for i in range(horizon):
            next_date = current_df.index[-1]+timedelta(days=1)
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
            X_df = pd.DataFrame([feat_dict])[feature_names].fillna(0)
            try:
                X_df[['lag_1','lag_7','rolling_mean_7','rolling_mean_14']] = scaler.transform(X_df[['lag_1','lag_7','rolling_mean_7','rolling_mean_14']])
            except:
                X_df_scaled = scaler.transform(X_df)
                X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)
            pred_log = model.predict(X_df)[0]
            pred_log = np.clip(pred_log,-10,15)
            pred_val = np.expm1(pred_log)*scenario_val
            pred_val *= (1+np.random.normal(0,noise_val))
            pred_val = np.clip(pred_val,0,MAX_SALES)
            bound = 1.96*metrics['residuals_std']*np.sqrt(i+1)
            preds.append(pred_val)
            lowers.append(max(0,pred_val-bound))
            uppers.append(min(MAX_SALES*1.2,pred_val+bound))
            current_df.loc[next_date]=[pred_val]
        forecasts[horizon] = {'dates':current_df.index[-horizon:], 'preds':preds,'low':lowers,'up':uppers}
    return forecasts

# ================== Run Forecast ==================
start_time = time.time()
forecasts = generate_forecast(df_store, horizons, scenario_val=1.0, noise_val=noise)
inf_time = time.time()-start_time

# ================== Dashboard UI ==================
st.title(f"📈 Retail Forecast | {selected_store}")

# ==== Metrics Display (Arabic/English) ====
m1,m2,m3,m4 = st.columns(4)
total_sales = sum([sum(f['preds']) for f in forecasts.values()])
m1.metric("إجمالي المبيعات المتوقع / Total Sales", f"${total_sales:,.0f}")
r2_display = f"{metrics['r2']*100:.1f}%" if metrics['r2']>0 else "غير كافٍ / Poor"
m2.metric("دقة التنبؤ (R²) / Forecast Accuracy", r2_display)
mape_display = f"{metrics['mape']*100:.1f}%" if np.isfinite(metrics['mape']) and metrics['mape']<=1 else "غير متاح / N/A"
m3.metric("خطأ التنبؤ (MAPE) / Prediction Error", mape_display)
m4.metric("زمن المعالجة / Inference Time", f"{inf_time*1000:.1f} ms")

# ==== Charts ====
for horizon in forecasts:
    f = forecasts[horizon]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_store.index[-max(horizons):], y=df_store['sales'].tail(max(horizons)), name="Actual", line=dict(color="#94a3b8")))
    fig.add_trace(go.Scatter(x=f['dates'], y=f['preds'], name=f"Forecast {horizon} يوم / days", line=dict(color="#3b82f6", width=4)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([f['dates'], f['dates'][::-1]]),
        y=np.concatenate([f['up'], f['low'][::-1]]),
        fill='toself', fillcolor='rgba(59,130,246,0.15)', line=dict(color='rgba(255,255,255,0)'), name="Confidence Interval"
    ))
    fig.update_layout(template=chart_template, paper_bgcolor=bg_color, plot_bgcolor=bg_color, hovermode="x unified", height=500)
    st.plotly_chart(fig,use_container_width=True)

# ==== Export Preview ====
for horizon in forecasts:
    f = forecasts[horizon]
    res_df = pd.DataFrame({"Date":f['dates'], "Forecast":f['preds'], "Min":f['low'], "Max":f['up']})
    st.subheader(f"Export Preview / معاينة التصدير ({horizon} يوم)")
    st.dataframe(res_df)
    st.download_button(f"Download CSV {horizon} Days", res_df.to_csv(index=False), f"forecast_{selected_store}_{horizon}.csv")

# ==== Feature Importance (Arabic Names) ====
st.subheader("🎯 أهم الميزات / Feature Importance")
feature_map = {
    'lag_1':'مبيعات اليوم السابق',
    'lag_7':'مبيعات الأسبوع السابق',
    'rolling_mean_7':'متوسط 7 أيام',
    'rolling_mean_14':'متوسط 14 يوم',
    'is_weekend':'عطلة نهاية الأسبوع',
    'was_closed_yesterday':'إغلاق أمس',
    'dayofweek_sin':'اليوم من الأسبوع (Sin)',
    'dayofweek_cos':'اليوم من الأسبوع (Cos)',
    'month_sin':'الشهر (Sin)',
    'month_cos':'الشهر (Cos)'
}
importance = model.get_feature_importance()
fig_imp = go.Figure(go.Bar(
    x=[importance[i] for i in range(len(feature_names))],
    y=[feature_map.get(feature_names[i],feature_names[i]) for i in range(len(feature_names))],
    orientation='h',
    marker=dict(color='#3b82f6')
))
fig_imp.update_layout(template=chart_template, height=400, paper_bgcolor=bg_color, plot_bgcolor=bg_color)
st.plotly_chart(fig_imp,use_container_width=True)

# ==== Footer ====
st.markdown("---")
st.markdown(f"""
<p style='text-align:center;font-size:14px;color:{text_color};'>
Developed by <strong>Eng. Goda Emad</strong> | 
<a href='https://github.com/Goda-Emad' target='_blank'>GitHub</a> | 
<a href='https://www.linkedin.com/in/goda-emad/' target='_blank'>LinkedIn</a>
</p>
""", unsafe_allow_html=True)


