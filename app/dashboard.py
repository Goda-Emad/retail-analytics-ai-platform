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
MODEL_VERSION = "v5.2 (Safe Mode)"
st.set_page_config(
    page_title=f"Retail AI {MODEL_VERSION}", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ================== 1️⃣ Language & Translation ==================
lang = st.sidebar.selectbox("🌐 اختر اللغة / Select Language", ["English", "عربي"])
def t(text_en, text_ar):
    return text_en if lang == "English" else text_ar

# ================== 2️⃣ Smart Assets Loader ==================
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
        st.error(f"❌ {t('Error loading files', 'خطأ في تحميل الملفات')}: {e}")
        return None, None, None, None

model, scaler, feature_names, df_raw = load_assets()
if model is None: st.stop()

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
            'is_weekend': 1 if next_date.dayofweek >= 5 else 0,
            'was_closed_yesterday': 1 if current_df['sales'].iloc[-1] <= 0 else 0
        }
        X_df = pd.DataFrame([feat_dict])[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        try:
            X_df[num_cols] = scaler.transform(X_df[num_cols])
        except:
            X_df_scaled = scaler.transform(X_df)
            X_df = pd.DataFrame(X_df_scaled, columns=feature_names, index=X_df.index)
        pred_log = np.clip(model.predict(X_df)[0], -10, 15)
        pred_val = np.expm1(pred_log) * scenario_val
        pred_val *= (1 + np.random.normal(0, noise_val))
        pred_val = np.clip(pred_val, 0, MAX_SALES)
        bound = 1.96 * residuals_std * np.sqrt(i+1)
        preds.append(pred_val)
        lowers.append(max(0, pred_val-bound))
        uppers.append(min(MAX_SALES*1.2, pred_val+bound))
        current_df.loc[next_date] = [pred_val]
    return preds, lowers, uppers, current_df.index[-horizon:]

# ================== 4️⃣ Sidebar UI ==================
st.sidebar.title(f"🚀 AI Retail Core {MODEL_VERSION}")
uploaded_file = st.sidebar.file_uploader(t("📂 Upload CSV","رفع ملف CSV"), type="csv")
df_active = process_upload(uploaded_file) if uploaded_file else df_raw.copy()
stores = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox(t("🏪 Select Store","اختر الفرع"), stores)
df_store = df_active[df_active['store_id']==selected_store] if 'store_id' in df_active.columns else df_active
if len(df_store)<30:
    st.error(t("⚠️ Not enough data (30 days required)","⚠️ بيانات غير كافية (30 يوم على الأقل)"))
    st.stop()
st.sidebar.divider()
horizon = st.sidebar.slider(t("Forecast Days","عدد أيام التنبؤ"),7,60,14)
scenario = st.sidebar.select_slider(t("Scenario","السيناريو"), options=[t("Pessimistic","متشائم"),t("Realistic","واقعي"),t("Optimistic","متفائل")], value=t("Realistic","واقعي"))
scenario_map = {t("Pessimistic","متشائم"):0.85, t("Realistic","واقعي"):1.0, t("Optimistic","متفائل"):1.15}
noise = st.sidebar.slider(t("Market Volatility","تقلب السوق"),0.0,0.2,0.05)

# ================== 5️⃣ Logic Execution ==================
@st.cache_data
def cached_backtesting(_df,_features,_scaler,_model):
    return run_backtesting(_df,_features,_scaler,_model)

metrics = cached_backtesting(df_store, feature_names, scaler, model)
start_inf = time.time()
preds, lowers, uppers, forecast_dates = generate_forecast(
    df_store,horizon,scenario_map[scenario],noise,metrics['residuals_std'],scaler,model,feature_names
)
inf_time = time.time()-start_inf

# ================== 6️⃣ Main Dashboard ==================
st.title(t(f"📈 Retail Forecast Dashboard | {selected_store}",
           f"📈 لوحة تحكم مبيعات التجزئة | {selected_store}"))

m1,m2,m3,m4=st.columns(4)
m1.metric(t("Expected Total Sales","إجمالي المبيعات المتوقع"), f"${np.sum(preds):,.0f}")
m2.metric(t("Model R² Score","دقة التنبؤ (R²)"), f"{metrics['r2']:.3f}")
m3.metric(t("Error Rate (MAPE)","خطأ التنبؤ (MAPE)"), f"{metrics['mape']*100:.2f}%")
m4.metric(t("Inference Time","زمن المعالجة"), f"{inf_time*1000:.1f} ms")

fig=go.Figure()
fig.add_trace(go.Scatter(x=df_store.index[-60:],y=df_store['sales'].tail(60),
                         name=t("Actual History","البيانات التاريخية"), line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=forecast_dates,y=preds,name=t("AI Safe Forecast","تنبؤ آمن"), line=dict(color="#3b82f6",width=4)))
fig.add_trace(go.Scatter(
    x=np.concatenate([forecast_dates,forecast_dates[::-1]]),
    y=np.concatenate([uppers,lowers[::-1]]),
    fill='toself', fillcolor='rgba(59,130,246,0.15)', line=dict(color='rgba(255,255,255,0)'), name=t("Confidence Interval","نطاق الثقة")
))
fig.update_layout(template="plotly_dark", hovermode="x unified", height=500, margin=dict(l=20,r=20,t=20,b=20))
st.plotly_chart(fig,use_container_width=True)

# ================== 7️⃣ Feature Importance ==================
col_a,col_b=st.columns(2)
with col_a:
    st.subheader(t("🎯 Feature Significance","🎯 أهم المتغيرات"))
    importance=model.get_feature_importance()
    feature_names_display = [
        t("Last Day Sales","مبيعات اليوم السابق"),
        t("Week Ago Sales","مبيعات قبل أسبوع"),
        t("7-Day Rolling Avg","متوسط 7 أيام"),
        t("14-Day Rolling Avg","متوسط 14 يوم"),
        t("Day of Week Sin","يوم الأسبوع Sin"),
        t("Day of Week Cos","يوم الأسبوع Cos"),
        t("Month Sin","الشهر Sin"),
        t("Month Cos","الشهر Cos"),
        t("Weekend Flag","عطلة نهاية الأسبوع"),
        t("Closed Yesterday","تم الإغلاق أمس")
    ]
    fig_imp=go.Figure(go.Bar(x=importance,y=feature_names_display,orientation='h',marker=dict(color='#3b82f6')))
    fig_imp.update_layout(template="plotly_dark",height=300,margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_imp,use_container_width=True)

with col_b:
    st.subheader(t("📥 Export & Preview","📥 تصدير وعرض"))
    res_df=pd.DataFrame({"Date":forecast_dates,"Forecast":preds,"Min":lowers,"Max":uppers})
    st.dataframe(res_df.head(10),use_container_width=True)
    st.download_button(t("Download CSV Report","تحميل التقرير CSV"), res_df.to_csv(index=False), f"forecast_{selected_store}.csv")

# ================== 8️⃣ Footer ==================
footer_html=f"""
<div style="text-align:center; font-size:14px; color:#888; margin-top:20px;">
<p>
👤 <strong>Eng. Goda Emad</strong> |
<a href='https://www.linkedin.com/in/goda-emad/' target='_blank'>
<img src='https://cdn-icons-png.flaticon.com/24/174/174857.png' style='vertical-align:middle;'> LinkedIn
</a> |
<a href='https://github.com/Goda-Emad' target='_blank'>
<img src='https://cdn-icons-png.flaticon.com/24/733/733553.png' style='vertical-align:middle;'> GitHub
</a> |
✉️ <a href='mailto:goda.emade2001@gmail.com'>goda.emade2001@gmail.com</a>
</p>
<p>
<img src='https://cdn-icons-png.flaticon.com/32/1828/1828817.png' width='32' style='vertical-align:middle;'> 
{t("Retail AI Dashboard","لوحة تحكم الذكاء الاصطناعي للتجزئة")}
</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
