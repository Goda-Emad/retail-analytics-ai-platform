# dashboard_v5_4.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os
import time
from utils import run_backtesting

# ================== 0️⃣ Config ==================
MODEL_VERSION = "v5.4 (Full Multi-Horizon)"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}",
                   layout="wide", page_icon="📈")

# ================== 1️⃣ Load Assets ==================
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

# ================== 2️⃣ Sidebar ==================
st.sidebar.title(f"🚀 Retail AI Control Center {MODEL_VERSION}")
theme_choice = st.sidebar.radio("🌗 Theme", ["Light", "Dark"])
st.markdown(f"<style>body {{background-color: {'#f9f9f9' if theme_choice=='Light' else '#0f1117'}};</style>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")
df_active = pd.read_csv(uploaded_file) if uploaded_file else df_raw.copy()
df_active.columns = [c.lower().strip() for c in df_active.columns]
if 'date' in df_active.columns:
    df_active['date'] = pd.to_datetime(df_active['date'])
    df_active = df_active.sort_values('date').set_index('date')

stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox("🏪 Select Store", stores)
df_store = df_active[df_active['store_id'] == selected_store] if 'store_id' in df_active.columns else df_active

if len(df_store) < 30:
    st.error("⚠️ بيانات غير كافية للتحليل (نحتاج 30 يوم على الأقل).")
    st.stop()

# Multi-Horizon Selection
horizons = st.sidebar.multiselect("⏱️ Select Forecast Horizons (days)", options=[7,14,30], default=[7,14,30])
scenario = st.sidebar.select_slider("Scenario", options=["متشائم","واقعي","متفائل"], value="واقعي")
scenario_map = {"متشائم":0.85, "واقعي":1.0, "متفائل":1.15}
noise = st.sidebar.slider("Market Volatility", 0.0, 0.2, 0.05)

# ================== 3️⃣ Cached Backtesting ==================
@st.cache_data
def cached_backtesting(_df, _features, _scaler, _model):
    return run_backtesting(_df, _features, _scaler, _model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 4️⃣ Forecast Function ==================
def generate_forecast(history_df, horizon, scenario_val, noise_val, residuals_std):
    preds, lowers, uppers = [], [], []
    current_df = history_df[['sales']].copy().replace([np.inf,-np.inf], np.nan).fillna(0)
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
            'was_closed_yesterday':1 if current_df['sales'].iloc[-1]<=0 else 0
        }
        X_df = pd.DataFrame([feat_dict])[feature_names].replace([np.inf,-np.inf],0).fillna(0)
        try:
            X_df[num_cols] = scaler.transform(X_df[num_cols])
        except:
            X_df_scaled = scaler.transform(X_df)
            X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)
        pred_log = model.predict(X_df)[0]
        pred_log = np.clip(pred_log,-10,15)
        pred_val = np.expm1(pred_log) * scenario_val
        pred_val *= (1+np.random.normal(0,noise_val))
        pred_val = np.clip(pred_val,0,MAX_SALES)
        preds.append(float(pred_val))
        bound = 1.96*residuals_std*np.sqrt(i+1)
        lowers.append(max(0,pred_val-bound))
        uppers.append(min(MAX_SALES*1.2,pred_val+bound))
        current_df.loc[next_date] = [pred_val]
    return preds, lowers, uppers, current_df.index[-horizon:]

# ================== 5️⃣ Run Forecast for All Horizons ==================
all_forecasts = {}
for h in horizons:
    preds, lowers, uppers, dates = generate_forecast(df_store,h,scenario_map[scenario],noise,metrics['residuals_std'])
    all_forecasts[h] = {"preds":preds,"lowers":lowers,"uppers":uppers,"dates":dates}

# ================== 6️⃣ Metrics Display ==================
st.title(f"📊 Retail Forecast Dashboard | {selected_store}")
for h in horizons:
    total_sales = np.sum(all_forecasts[h]['preds'])
    st.metric(label=f"⏱️ إجمالي المبيعات المتوقع ({h} يوم)", value=f"${total_sales:,.0f}")

st.metric(label="📈 دقة التنبؤ (R²)", value=f"{metrics['r2']:.3f}")
st.metric(label="⚠️ خطأ التنبؤ (MAPE)", value=f"{metrics['mape']*100:.2f}%")
st.metric(label="⏱️ زمن المعالجة", value=f"{metrics['execution_time']*1000:.1f} ms")

# ================== 7️⃣ Multi-Horizon Chart ==================
fig = go.Figure()
colors = ["#3b82f6","#f97316","#16a34a"]
for idx,h in enumerate(horizons):
    fig.add_trace(go.Scatter(x=all_forecasts[h]['dates'], y=all_forecasts[h]['preds'],
                             name=f"{h}-day Forecast", line=dict(color=colors[idx%len(colors)],width=4)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([all_forecasts[h]['dates'],all_forecasts[h]['dates'][::-1]]),
        y=np.concatenate([all_forecasts[h]['uppers'],all_forecasts[h]['lowers'][::-1]]),
        fill='toself', fillcolor='rgba(59,130,246,0.15)', line=dict(color='rgba(255,255,255,0)'), showlegend=False
    ))
fig.add_trace(go.Scatter(x=df_store.index[-60:], y=df_store['sales'].tail(60), name="Actual Sales", line=dict(color="#94a3b8")))
fig.update_layout(template="plotly_dark" if theme_choice=="Dark" else "plotly_white",
                  hovermode="x unified", height=500, margin=dict(l=20,r=20,t=20,b=20))
st.plotly_chart(fig,use_container_width=True)

# ================== 8️⃣ Feature Importance ==================
feat_names_display = {
    'lag_1':'مبيعات اليوم السابق',
    'lag_7':'مبيعات الأسبوع السابق',
    'rolling_mean_7':'متوسط 7 أيام',
    'rolling_mean_14':'متوسط 14 يوم',
    'dayofweek_sin':'اليوم الأسبوعي (Sine)',
    'dayofweek_cos':'اليوم الأسبوعي (Cos)',
    'month_sin':'الشهر (Sine)',
    'month_cos':'الشهر (Cos)',
    'is_weekend':'عطلة نهاية الأسبوع',
    'was_closed_yesterday':'إغلاق أمس'
}
importance = model.get_feature_importance()
importance_names = [feat_names_display.get(f,f) for f in feature_names]
fig_imp = go.Figure(go.Bar(x=importance, y=importance_names, orientation='h', marker=dict(color='#3b82f6')))
fig_imp.update_layout(template="plotly_dark" if theme_choice=="Dark" else "plotly_white",
                      height=300, margin=dict(l=20,r=20,t=40,b=20))
st.subheader("🎯 Feature Significance / أهمية الميزات")
st.plotly_chart(fig_imp,use_container_width=True)

# ================== 9️⃣ Export Preview ==================
export_df = pd.DataFrame()
for h in horizons:
    df_h = pd.DataFrame({"Date":all_forecasts[h]['dates'],
                         f"Forecast_{h}":all_forecasts[h]['preds'],
                         f"Min_{h}":all_forecasts[h]['lowers'],
                         f"Max_{h}":all_forecasts[h]['uppers']})
    export_df = pd.concat([export_df, df_h.set_index("Date")], axis=1)
export_df.reset_index(inplace=True)
st.subheader("📥 Export Preview / معاينة التصدير")
st.dataframe(export_df.head(10),use_container_width=True)
st.download_button("Download CSV / تحميل CSV", export_df.to_csv(index=False), f"forecast_{selected_store}.csv")

# ================== 10️⃣ Footer ==================
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;">
    <div>Eng. Goda Emad</div>
    <div>
        <a href="https://github.com/Goda-Emad" target="_blank" style="margin-right:10px;">GitHub</a>
        <a href="https://www.linkedin.com/in/goda-emad/" target="_blank">LinkedIn</a>
    </div>
    <div>© 2026 Retail AI Dashboard</div>
</div>
""", unsafe_allow_html=True)
