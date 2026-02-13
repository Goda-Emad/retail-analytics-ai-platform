import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os
import time
from utils import run_backtesting

# ================== 0️⃣ الإعدادات والثيم (Dark/Light) ==================
MODEL_VERSION = "v5.5 (Pro Ultra)"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# اختيار الثيم والخلفية
st.sidebar.title("🎨 مظهر المنصة")
theme_choice = st.sidebar.selectbox("اختر وضع العرض", options=["Dark Mode", "Light Mode"])

if theme_choice == "Dark Mode":
    bg_css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    [data-testid="stSidebar"] {background-color: rgba(15, 23, 42, 0.8);}
    </style>
    """
    chart_template = "plotly_dark"
    neon_color = "#00f2fe" # أزرق نيون
else:
    bg_css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #1e293b;
    }
    </style>
    """
    chart_template = "plotly_white"
    neon_color = "#3b82f6" # أزرق ملكي

st.markdown(bg_css, unsafe_allow_html=True)

# ================== 1️⃣ تحميل الملفات (كودك الأصلي) ==================
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

# ================== 2️⃣ معالجة البيانات واللغة ==================
lang_choice = st.sidebar.selectbox("🌐 اللغة / Language", options=["عربي", "English"])
def t(ar, en): return ar if lang_choice=="عربي" else en

uploaded_file = st.sidebar.file_uploader(t("رفع ملف CSV", "Upload CSV"), type="csv")
df_active = pd.read_csv(uploaded_file) if uploaded_file else df_raw.copy()
df_active.columns = [c.lower().strip() for c in df_active.columns]
if 'date' in df_active.columns:
    df_active['date'] = pd.to_datetime(df_active['date'])
    df_active = df_active.sort_values('date').set_index('date')

stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox(t("اختر المتجر", "Select Store"), stores)
df_store = df_active[df_active['store_id']==selected_store] if 'store_id' in df_active.columns else df_active

forecast_horizon = st.sidebar.slider(t("أيام التوقع", "Forecast Days"), 1, 60, 14)
scenario = st.sidebar.select_slider(t("السيناريو", "Scenario"), options=["متشائم","واقعي","متفائل"], value="واقعي")
scenario_map = {"متشائم":0.85,"واقعي":1.0,"متفائل":1.15}

@st.cache_data
def cached_backtesting(_df, _f, _s, _m): return run_backtesting(_df, _f, _s, _m)
metrics = cached_backtesting(df_store, feature_names, scaler, model)

# ================== 3️⃣ دالة التوقع (Logic) ==================
def generate_forecast(history_df, horizon, scenario_val, residuals_std):
    np.random.seed(42)
    preds, lowers, uppers = [], [], []
    current_df = history_df[['sales']].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    num_cols = ['lag_1','lag_7','rolling_mean_7','rolling_mean_14']
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
        X_df = pd.DataFrame([feat_dict])[feature_names]
        X_scaled = scaler.transform(X_df)
        pred_log = model.predict(X_scaled)[0]
        pred_val = np.expm1(np.clip(pred_log, -10, 15)) * scenario_val
        bound = 1.96 * residuals_std * np.sqrt(i+1)
        preds.append(pred_val); lowers.append(max(0, pred_val-bound)); uppers.append(pred_val+bound)
        current_df.loc[next_date] = [pred_val]
    return preds, lowers, uppers, current_df.index[-horizon:]

preds, lowers, uppers, forecast_dates = generate_forecast(df_store, forecast_horizon, scenario_map[scenario], metrics['residuals_std'])

# ================== 4️⃣ العرض البصري (Charts & Tables) ==================
st.title(f"📈 {t('توقعات المبيعات الذكية', 'Smart Sales Forecast')} | {selected_store}")

# المخطط البياني بالألوان المطلوبة
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-60:], y=df_store['sales'].tail(60), name=t("التاريخي", "Actual"), line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=forecast_dates, y=preds, name=t("توقع الذكاء الاصطناعي", "AI Forecast"), line=dict(color=neon_color, width=4)))
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
    y=np.concatenate([uppers, lowers[::-1]]),
    fill='toself', fillcolor='rgba(0, 242, 254, 0.1)', line=dict(color='rgba(255,255,255,0)'), name=t("نطاق الثقة", "Confidence")
))
fig.update_layout(template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader(t("🎯 أهم العوامل المؤثرة", "🎯 Key Features"))
    importance = model.get_feature_importance()
    feature_map_ar = {
        'lag_1': "مبيعات اليوم السابق", 'lag_7': "مبيعات الأسبوع الماضي",
        'rolling_mean_7': "متوسط آخر 7 أيام", 'rolling_mean_14': "متوسط آخر 14 يوم",
        'is_weekend': "عطلة نهاية الأسبوع", 'was_closed_yesterday': "إغلاق أمس"
    }
    display_names = [feature_map_ar.get(n, n) for n in feature_names] if lang_choice=="عربي" else feature_names
    fig_imp = go.Figure(go.Bar(x=importance, y=display_names, orientation='h', marker=dict(color=neon_color)))
    fig_imp.update_layout(template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

with col2:
    st.subheader(t("📥 معاينة البيانات", "📥 Data Preview"))
    res_df = pd.DataFrame({"Date": forecast_dates, "Forecast": preds, "Min": lowers, "Max": uppers})
    # الجدول الملون بتدرج الأزرق
    st.dataframe(res_df.style.background_gradient(cmap='Blues', subset=['Forecast']), use_container_width=True)
    st.download_button(t("تحميل التقرير", "Download CSV"), res_df.to_csv(index=False), "forecast.csv")

# ================== 5️⃣ Footer (كودك الأصلي) ==================
st.markdown("---")
st.markdown(f"<div style='text-align:center; opacity:0.7;'>Eng. Goda Emad | v{MODEL_VERSION} | 2026</div>", unsafe_allow_html=True)
