import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import time
import os
from utils import run_backtesting

# ================== 0️⃣ Config ==================
MODEL_VERSION = "v5.4 (Multi-Horizon & Bilingual)"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# ================== 1️⃣ Assets Loader ==================
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

# ================== 2️⃣ Language & Theme ==================
lang_choice = st.sidebar.selectbox("🌐 اللغة / Language", ["English", "عربي"])
theme_choice = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])
st.markdown(f"<style>body {{ background-color: {'#f9f9f9' if theme_choice=='Light' else '#0f1117'}; }}</style>", unsafe_allow_html=True)

# Translation dictionary
texts = {
    "English": {
        "upload": "📂 Upload CSV",
        "select_store": "🏪 Select Store",
        "forecast_days": "Forecast Days",
        "scenario": "Scenario",
        "market_vol": "Market Volatility",
        "expected_sales": "Expected Total Sales",
        "r2_score": "Prediction Accuracy (R²)",
        "mape": "Prediction Error (MAPE)",
        "inference": "Inference Time",
        "feature_imp": "🎯 Feature Importance",
        "export_preview": "📥 Export Preview",
        "system_diag": "📝 System Diagnostics",
        "safe_mode": "Safe Mode Active"
    },
    "عربي": {
        "upload": "📂 رفع ملف CSV",
        "select_store": "🏪 اختر المتجر",
        "forecast_days": "عدد أيام التوقع",
        "scenario": "السيناريو",
        "market_vol": "تقلب السوق",
        "expected_sales": "إجمالي المبيعات المتوقع",
        "r2_score": "دقة التنبؤ (R²)",
        "mape": "خطأ التنبؤ (MAPE)",
        "inference": "زمن المعالجة",
        "feature_imp": "🎯 أهم الميزات",
        "export_preview": "📥 معاينة التصدير",
        "system_diag": "📝 تشخيص النظام",
        "safe_mode": "الوضع الآمن مفعل"
    }
}

t = texts[lang_choice]

# ================== 3️⃣ Core Functions ==================
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

# ================== 4️⃣ Sidebar ==================
st.sidebar.title(f"🚀 AI Retail Core {MODEL_VERSION}")
uploaded_file = st.sidebar.file_uploader(t["upload"], type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()
stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox(t["select_store"], stores)
df_store = df_active[df_active['store_id'] == selected_store] if 'store_id' in df_active.columns else df_active
if len(df_store) < 30:
    st.error("⚠️ بيانات غير كافية للتحليل (نحتاج 30 يوم عمل على الأقل).")
    st.stop()

horizon_days = st.sidebar.multiselect(t["forecast_days"], options=[7,14,30,60,90], default=[14,30])
scenario = st.sidebar.select_slider(t["scenario"], options=["متشائم","واقعي","متفائل"], value="واقعي")
scenario_map = {"متشائم":0.85,"واقعي":1.0,"متفائل":1.15}
noise = st.sidebar.slider(t["market_vol"], 0.0, 0.2, 0.05)

# ================== 5️⃣ Backtesting ==================
@st.cache_data
def cached_backtesting(_df, _features, _scaler, _model):
    return run_backtesting(_df, _features, _scaler, _model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 6️⃣ Forecast Execution ==================
all_forecasts = {}
for h in horizon_days:
    preds, lowers, uppers, forecast_dates = generate_forecast(
        df_store, h, scenario_map[scenario], noise, metrics['residuals_std'],
        scaler, model, feature_names
    )
    all_forecasts[h] = {"preds":preds,"lowers":lowers,"uppers":uppers,"dates":forecast_dates}

# ================== 7️⃣ Dashboard ==================
st.title(f"📈 Retail Forecast | {selected_store}")

# Metrics cards (Multi-Horizon)
cols = st.columns(len(horizon_days))
for idx,h in enumerate(horizon_days):
    total_sales = np.sum(all_forecasts[h]["preds"])
    cols[idx].metric(f"{t['expected_sales']} ({h} يوم)", f"${total_sales:,.0f}")

st.subheader(t["r2_score"])
st.write(f"{metrics['r2']:.3f}")
st.subheader(t["mape"])
st.write(f"{metrics['mape']*100:.2f}%")
st.subheader(t["inference"])
st.write(f"{0.001*metrics['execution_time']:.2f} s")  # ms -> s

# ================== 8️⃣ Forecast Chart ==================
fig = go.Figure()
for h in horizon_days:
    fig.add_trace(go.Scatter(
        x=all_forecasts[h]["dates"], 
        y=all_forecasts[h]["preds"],
        name=f"{h}-Day Forecast",
        line=dict(width=3)
    ))
fig.add_trace(go.Scatter(
    x=df_store.index[-60:], 
    y=df_store['sales'].tail(60), 
    name="Actual History", 
    line=dict(color="#94a3b8")
))
fig.update_layout(template="plotly_dark" if theme_choice=="Dark" else "plotly_white", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# ================== 9️⃣ Feature Importance ==================
st.subheader(t["feature_imp"])
feat_map = {
    "lag_1":"مبيعات أمس / Yesterday Sales",
    "lag_7":"مبيعات الأسبوع الماضي / Last Week Sales",
    "rolling_mean_7":"متوسط 7 أيام / 7-Day Avg",
    "rolling_mean_14":"متوسط 14 يوم / 14-Day Avg",
    "is_weekend":"إجازة / Weekend",
    "was_closed_yesterday":"أغلق أمس / Closed Yesterday"
}
importance = model.get_feature_importance()
names = [feat_map.get(f,f) for f in feature_names]
fig_imp = go.Figure(go.Bar(x=importance, y=names, orientation='h', marker=dict(color='#3b82f6')))
fig_imp.update_layout(template="plotly_dark" if theme_choice=="Dark" else "plotly_white", height=350)
st.plotly_chart(fig_imp, use_container_width=True)

# ================== 10️⃣ Export Preview ==================
st.subheader(t["export_preview"])
for h in horizon_days:
    res_df = pd.DataFrame({
        "Date": all_forecasts[h]["dates"],
        "Forecast": all_forecasts[h]["preds"],
        "Min": all_forecasts[h]["lowers"],
        "Max": all_forecasts[h]["uppers"]
    })
    st.write(f"{h}-Day Forecast")
    st.dataframe(res_df.head(10), use_container_width=True)
    st.download_button(f"Download {h}-Day CSV", res_df.to_csv(index=False), f"forecast_{selected_store}_{h}d.csv")

# ================== 11️⃣ Footer ==================
st.markdown("---")
st.markdown(f"""
<p style="text-align:center; font-size:14px;">
Eng. Goda Emad | 
<a href='https://github.com/Goda-Emad' target='_blank'>GitHub</a> | 
<a href='https://www.linkedin.com/in/goda-emad/' target='_blank'>LinkedIn</a> | 
© 2026
</p>
""", unsafe_allow_html=True)
st.markdown("---")

