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
MODEL_VERSION = "v5.3 (Heatmap CI Modern)"
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

# ================== 2️⃣ Core Functions ==================
def process_upload(file):
    uploaded_df = pd.read_csv(file)
    uploaded_df.columns = [c.lower().strip() for c in uploaded_df.columns]
    if 'date' in uploaded_df.columns:
        uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
        uploaded_df = uploaded_df.sort_values('date').set_index('date')
    return uploaded_df

def generate_forecast(history_df, horizon, scenario_val, noise_val, residuals_std, scaler, model, feature_names):
    np.random.seed(42)
    preds, lowers, uppers = [], [], []
    
    current_df = history_df[['sales']].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    MAX_SALES = max(100000, current_df['sales'].max() * 3)

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
            'is_weekend': 1 if next_date.dayofweek >= 5 else 0,
            'was_closed_yesterday': 1 if current_df['sales'].iloc[-1] <= 0 else 0
        }
        X_df = pd.DataFrame([feat_dict])[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        try:
            X_df[num_cols] = scaler.transform(X_df[num_cols])
        except:
            X_df_scaled = scaler.transform(X_df)
            X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)

        pred_log = model.predict(X_df)[0]
        pred_log = np.clip(pred_log, -10, 15)
        pred_val = np.expm1(pred_log) * scenario_val
        pred_val *= (1 + np.random.normal(0, noise_val))
        pred_val = np.clip(pred_val, 0, MAX_SALES)
        pred_val = float(pred_val)

        bound = 1.96 * residuals_std * np.sqrt(i + 1)
        preds.append(pred_val)
        lowers.append(max(0, pred_val - bound))
        uppers.append(min(MAX_SALES*1.2, pred_val + bound))
        current_df.loc[next_date] = [pred_val]

    return preds, lowers, uppers, current_df.index[-horizon:]

# ================== 3️⃣ Sidebar UI ==================
st.sidebar.title(f"🚀 AI Retail Core {MODEL_VERSION}")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()
stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox("🏪 Select Store", stores)
df_store = df_active[df_active['store_id'] == selected_store] if 'store_id' in df_active.columns else df_active
if len(df_store) < 30: st.error("⚠️ بيانات غير كافية للتحليل (نحتاج 30 يوم عمل على الأقل)."); st.stop()
st.sidebar.divider()
horizons = st.sidebar.multiselect("Select Forecast Horizons (Days)", options=[7,14,30], default=[7,14,30])
scenario = st.sidebar.select_slider("Scenario", options=["متشائم","واقعي","متفائل"], value="واقعي")
scenario_map = {"متشائم":0.85,"واقعي":1.0,"متفائل":1.15}
noise = st.sidebar.slider("Market Volatility", 0.0,0.2,0.05)

# ================== 4️⃣ Backtesting ==================
@st.cache_data
def cached_backtesting(_df, _features, _scaler, _model):
    return run_backtesting(_df, _features, _scaler, _model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 5️⃣ Forecast for Multiple Horizons ==================
forecast_results = {}
start_inf = time.time()
for h in horizons:
    preds, lowers, uppers, dates = generate_forecast(df_store, h, scenario_map[scenario], noise, metrics['residuals_std'], scaler, model, feature_names)
    forecast_results[h] = {'preds': preds, 'lowers': lowers, 'uppers': uppers, 'dates': dates}
inf_time = time.time() - start_inf

# ================== 6️⃣ Main Dashboard UI ==================
st.title(f"📈 Retail Forecast Dashboard | {selected_store}")
m1, m2, m3, m4 = st.columns(4)
total_forecast = sum([sum(forecast_results[h]['preds']) for h in horizons])
m1.metric("Expected Total Sales", f"${total_forecast:,.0f}")
m2.metric("Model R² Score", f"{metrics['r2']:.3f}")
m3.metric("Error Rate (MAPE)", f"{metrics['mape']*100:.2f}%")
m4.metric("Inference Time", f"{inf_time*1000:.1f} ms")

# ================== 7️⃣ Forecast Chart + CI ==================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-60:], y=df_store['sales'].tail(60), name="Actual History", line=dict(color="#94a3b8")))
for h in horizons:
    f = forecast_results[h]
    fig.add_trace(go.Scatter(x=f['dates'], y=f['preds'], name=f"{h}-Day Forecast", line=dict(width=3)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([f['dates'], f['dates'][::-1]]),
        y=np.concatenate([f['uppers'], f['lowers'][::-1]]),
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f"{h}D CI"
    ))
fig.update_layout(template="plotly_dark", hovermode="x unified", height=500, margin=dict(l=20,r=20,t=20,b=20))
st.plotly_chart(fig, use_container_width=True)

# ================== 8️⃣ Dynamic Heatmap with CI ==================
st.subheader("🔥 Forecast Heatmap + CI")
if horizons:
    max_h = max(forecast_results.keys(), key=lambda h: len(forecast_results[h]['dates']))
    heatmap_index = forecast_results[max_h]['dates']
    heatmap_df = pd.DataFrame({
        h: pd.Series(forecast_results[h]['preds'], index=forecast_results[h]['dates']).reindex(heatmap_index, fill_value=np.nan)
        for h in horizons
    })
    heatmap_lower = pd.DataFrame({
        h: pd.Series(forecast_results[h]['lowers'], index=forecast_results[h]['dates']).reindex(heatmap_index, fill_value=np.nan)
        for h in horizons
    })
    heatmap_upper = pd.DataFrame({
        h: pd.Series(forecast_results[h]['uppers'], index=forecast_results[h]['dates']).reindex(heatmap_index, fill_value=np.nan)
        for h in horizons
    })
    colorscale = 'Viridis'
    fig_heatmap = go.Figure()
    for idx, h in enumerate(horizons):
        z = heatmap_df[h].values
        z_lower = heatmap_lower[h].values
        z_upper = heatmap_upper[h].values
        fig_heatmap.add_trace(go.Heatmap(
            z=[z_lower, z, z_upper],
            x=heatmap_index,
            y=[f"{h}D Lower", f"{h}D Forecast", f"{h}D Upper"],
            colorscale=colorscale,
            showscale=False
        ))
    fig_heatmap.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ================== 9️⃣ Feature Importance + Export ==================
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🎯 Feature Significance")
    importance = model.get_feature_importance()
    fig_imp = go.Figure(go.Bar(x=importance, y=feature_names, orientation='h', marker=dict(color='#3b82f6')))
    fig_imp.update_layout(template="plotly_dark", height=300, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

with col_b:
    st.subheader("📥 Export Preview")
    export_dates = heatmap_index
    export_df = pd.DataFrame({
        f"{h}D Forecast": pd.Series(forecast_results[h]['preds'], index=forecast_results[h]['dates']).reindex(export_dates, fill_value=np.nan)
        for h in horizons
    })
    export_df['Date'] = export_dates
    st.dataframe(export_df.head(10), use_container_width=True)
    st.download_button("Download CSV Report", export_df.to_csv(index=False), f"forecast_{selected_store}.csv")

# ================== 10️⃣ Diagnostics ==================
with st.expander("📝 System Diagnostics"):
    st.write(f"**Version:** {MODEL_VERSION}")
    st.write(f"**Safe Mode:** Active (MAX_SALES Cap)")
    st.write(f"**Backtesting Samples:** {metrics['data_points']}")
    st.write(f"**Residual Std Dev:** {metrics['residuals_std']:.2f}")

