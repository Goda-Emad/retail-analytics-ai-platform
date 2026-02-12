import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import joblib, os, time
from utils import run_backtesting

# ================== 0️⃣ Config ==================
MODEL_VERSION = "v5.0 Pro Enterprise"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="🚀")

# ================== 1️⃣ Load Assets ==================
@st.cache_resource
def load_assets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        paths = {
            "model": os.path.join(current_dir,"catboost_sales_model_10features.pkl"),
            "scaler": os.path.join(current_dir,"scaler_10features.pkl"),
            "features": os.path.join(current_dir,"feature_names_10features.pkl"),
            "data": os.path.join(current_dir,"daily_sales_ready_10features.parquet")
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
        st.error(f"❌ Error loading assets: {e}")
        return None,None,None,None

model, scaler, feature_names, df_raw = load_assets()
if model is None: st.stop()

# ================== 2️⃣ Helpers ==================
def process_upload(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
    return df

def generate_forecast(df_store, horizon, scenario_val, noise_val, residuals_std):
    np.random.seed(42)
    preds, lowers, uppers = [], [], []
    alerts = []
    current_df = df_store[['sales']].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    num_cols=['lag_1','lag_7','rolling_mean_7','rolling_mean_14']
    MAX_SALES = max(100000, current_df['sales'].max()*3)

    for i in range(horizon):
        next_date = current_df.index[-1] + timedelta(days=1)
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
        X_df = pd.DataFrame([feat_dict])[feature_names].replace([np.inf,-np.inf],np.nan).fillna(0)
        try: X_df[num_cols] = scaler.transform(X_df[num_cols])
        except:
            X_df_scaled = scaler.transform(X_df)
            X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)

        pred_log = np.clip(model.predict(X_df)[0], -10, 15)
        pred_val = np.clip(np.expm1(pred_log) * scenario_val * (1 + np.random.normal(0, noise_val)), 0, MAX_SALES)
        if pred_val >= MAX_SALES*0.95: alerts.append((next_date,pred_val))
        bound = 1.96 * residuals_std * np.sqrt(i+1)
        preds.append(pred_val)
        lowers.append(max(0, pred_val-bound))
        uppers.append(min(MAX_SALES*1.2, pred_val+bound))
        current_df.loc[next_date] = [pred_val]

    return preds, lowers, uppers, current_df.index[-horizon:], alerts

# ================== 3️⃣ Sidebar ==================
st.sidebar.title(f"🚀 Retail AI {MODEL_VERSION}")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()
stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox("🏪 Select Store", stores)
df_store = df_active[df_active['store_id']==selected_store] if 'store_id' in df_active.columns else df_active
if len(df_store)<30: st.error("⚠️ Not enough data"); st.stop()

horizons = [7,14,30]
scenario_dynamic = st.sidebar.select_slider("Scenario", ["متشائم","واقعي","متفائل"], value="واقعي")
noise_dynamic = st.sidebar.slider("Market Volatility",0.0,0.2,0.05)

# ================== 4️⃣ Backtesting ==================
@st.cache_data
def cached_backtesting(_df,_features,_scaler,_model):
    return run_backtesting(_df,_features,_scaler,_model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 5️⃣ Forecasts ==================
scenario_map = {"متشائم":0.85,"واقعي":1.0,"متفائل":1.15}
forecast_data = {}
alerts_store = []
for h in horizons:
    preds, lowers, uppers, dates, alerts = generate_forecast(df_store,h,scenario_map[scenario_dynamic],noise_dynamic,metrics['residuals_std'])
    forecast_data[h] = {'preds': preds,'lowers': lowers,'uppers': uppers,'dates': dates}
    alerts_store.extend(alerts)

# ================== 6️⃣ Heatmap ==================
heat_df = pd.DataFrame()
for h in horizons:
    heat_df[f"{h}-day Forecast"] = forecast_data[h]['preds']
heat_df.index = forecast_data[horizons[0]]['dates']
fig_heat = px.imshow(heat_df.T, text_auto=True, color_continuous_scale='blues')
fig_heat.update_layout(title="Forecast Heatmap per Horizon",template="plotly_dark",height=400)
st.plotly_chart(fig_heat,use_container_width=True)

# ================== 7️⃣ Alerts ==================
if alerts_store:
    st.warning(f"⚠️ Forecast close to MAX_SALES on dates: {', '.join([str(a[0].date()) for a in alerts_store])}")
    # يمكن هنا إضافة Slack/Email API لإرسال التحذيرات تلقائياً

# ================== 8️⃣ Charts ==================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-60:],y=df_store['sales'].tail(60),name="History",line=dict(color="#94a3b8")))
colors=["#3b82f6","#10b981","#f59e0b"]
for idx,h in enumerate(horizons):
    data=forecast_data[h]
    fig.add_trace(go.Scatter(x=data['dates'],y=data['preds'],name=f"{h}-day Forecast",line=dict(width=3,color=colors[idx%len(colors)])))
    fig.add_trace(go.Scatter(x=np.concatenate([data['dates'],data['dates'][::-1]]),
                             y=np.concatenate([data['uppers'],data['lowers'][::-1]]),
                             fill='toself',fillcolor='rgba(59,130,246,0.15)',
                             line=dict(color='rgba(255,255,255,0)'),showlegend=False))
fig.update_layout(template="plotly_dark",hovermode="x unified",height=550)
st.plotly_chart(fig,use_container_width=True)

# ================== 9️⃣ Feature Importance ==================
col_a,col_b=st.columns(2)
with col_a:
    st.subheader("🎯 Feature Importance")
    importance = model.get_feature_importance()
    fig_imp = go.Figure(go.Bar(x=importance,y=feature_names,orientation='h',marker=dict(color="#3b82f6")))
    fig_imp.update_layout(template="plotly_dark",height=300)
    st.plotly_chart(fig_imp,use_container_width=True)

# ================== 10️⃣ Export CSV ==================
with col_b:
    st.subheader("📥 Export CSV")
    for h in horizons:
        df_exp=pd.DataFrame({"Date":forecast_data[h]['dates'], "Forecast":forecast_data[h]['preds']})
        st.download_button(f"Download {h}-day Forecast CSV",df_exp.to_csv(index=False),f"forecast_{selected_store}_{h}days.csv")

# ================== 11️⃣ Diagnostics ==================
with st.expander("📝 System Diagnostics"):
    st.write(f"Residual Std Dev: {metrics['residuals_std']:.2f}")
    st.write(f"Backtesting Samples: {metrics['data_points']}")
    st.write(f"Safe Mode MAX Cap Applied ✅")
