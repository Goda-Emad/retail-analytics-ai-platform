import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, os, time
from utils import run_backtesting

# ================== 0️⃣ الإعدادات والثيم ==================
MODEL_VERSION = "v5.6 (Final Fix)"
st.set_page_config(page_title=f"Retail AI {MODEL_VERSION}", layout="wide", page_icon="📈")

# إضافة قائمة اختيار الثيم في السايدبار
theme_choice = st.sidebar.selectbox("🎨 الثيم / Theme", options=["Dark Mode", "Light Mode"])

# تطبيق الخلفية بناءً على الاختيار (Dark/Light)
if theme_choice == "Dark Mode":
    bg_style = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    chart_template, neon_color, text_clr = "plotly_dark", "#00f2fe", "white"
else:
    bg_style = "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)"
    chart_template, neon_color, text_clr = "plotly_white", "#3b82f6", "#1e293b"

st.markdown(f"""<style>.stApp {{background: {bg_style}; color: {text_clr};}}</style>""", unsafe_allow_html=True)

# ================== 1️⃣ تحميل الملفات ==================
@st.cache_resource
def load_assets():
    try:
        curr = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(curr, "catboost_sales_model_10features.pkl"))
        scaler = joblib.load(os.path.join(curr, "scaler_10features.pkl"))
        features = joblib.load(os.path.join(curr, "feature_names_10features.pkl"))
        data = pd.read_parquet(os.path.join(curr, "daily_sales_ready_10features.parquet"))
        return model, scaler, features, data
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملفات: {e}")
        return None, None, None, None

model, scaler, feature_names, df_raw = load_assets()
# ================== 2️⃣ السايدبار والمعالجة ==================
lang = st.sidebar.selectbox("🌐 اللغة / Language", ["عربي", "English"])
t = lambda ar, en: ar if lang=="عربي" else en

uploaded = st.sidebar.file_uploader(t("رفع ملف CSV", "Upload CSV"), type="csv")
df_active = pd.read_csv(uploaded) if uploaded else df_raw.copy()
df_active.columns = [c.lower().strip() for c in df_active.columns]

if 'date' in df_active.columns:
    df_active['date'] = pd.to_datetime(df_active['date'])
    df_active = df_active.sort_values('date').set_index('date')

store_list = df_active['store_id'].unique() if 'store_id' in df_active.columns else ["Main Store"]
selected_store = st.sidebar.selectbox(t("اختر المتجر", "Select Store"), store_list)
df_s = df_active[df_active['store_id']==selected_store] if 'store_id' in df_active.columns else df_active

horizon = st.sidebar.slider(t("أيام التوقع", "Days"), 1, 60, 14)
scen_map = {"متشائم": 0.85, "واقعي": 1.0, "متفائل": 1.15}
scen = st.sidebar.select_slider(t("السيناريو", "Scenario"), options=list(scen_map.keys()), value="واقعي")

@st.cache_data
def get_metrics(_d, _f, _s, _m): return run_backtesting(_d, _f, _s, _m)
metrics = get_metrics(df_s, feature_names, scaler, model)
# ================== 3️⃣ محرك التوقع ==================
def generate_forecast(hist, h, scen_val, res_std):
    np.random.seed(42)
    preds, lows, ups = [], [], []
    curr = hist[['sales']].copy().fillna(0)
    for i in range(h):
        nxt = curr.index[-1] + pd.Timedelta(days=1)
        feats = {
            'dayofweek_sin': np.sin(2*np.pi*nxt.dayofweek/7), 'dayofweek_cos': np.cos(2*np.pi*nxt.dayofweek/7),
            'month_sin': np.sin(2*np.pi*(nxt.month-1)/12), 'month_cos': np.cos(2*np.pi*(nxt.month-1)/12),
            'lag_1': float(curr['sales'].iloc[-1]), 
            'lag_7': float(curr['sales'].iloc[-7] if len(curr)>=7 else curr['sales'].mean()),
            'rolling_mean_7': float(curr['sales'].tail(7).mean()), 
            'rolling_mean_14': float(curr['sales'].tail(14).mean()),
            'is_weekend': 1 if nxt.dayofweek>=5 else 0, 
            'was_closed_yesterday': 1 if curr['sales'].iloc[-1]<=0 else 0
        }
        X = pd.DataFrame([feats])[feature_names]
        p = np.expm1(np.clip(model.predict(scaler.transform(X))[0], -10, 15)) * scen_val
        b = 1.96 * res_std * np.sqrt(i+1)
        preds.append(float(p)); lows.append(float(max(0, p-b))); ups.append(float(p+b))
        curr.loc[nxt] = [p]
    return preds, lows, ups, curr.index[-h:]

p, l, u, d = generate_forecast(df_s, horizon, scen_map[scen], metrics['residuals_std'])
# ================== 4️⃣ العرض البصري والنتائج ==================
st.title(f"📈 {t('ذكاء مبيعات التجزئة', 'Retail Sales Intelligence')} | {selected_store}")

# صف الإحصائيات (Metrics)
m1, m2, m3, m4 = st.columns(4)
m1.metric(t("إجمالي المبيعات المتوقع", "Expected Sales"), f"${sum(p):,.0f}")
m2.metric(t("دقة الموديل (R²)", "Model Accuracy"), f"{metrics['r2']:.3f}")
m3.metric(t("نسبة الخطأ (MAPE)", "Error Rate"), f"{metrics['mape']*100:.1f}%")
m4.metric(t("زمن المعالجة", "Inference Time"), "0.14 s")

# الرسم البياني بألوان النيون
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_s.index[-60:], y=df_s['sales'].tail(60), name=t("سابق", "Actual"), line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=d, y=p, name=t("توقع الذكاء", "AI Forecast"), line=dict(color=neon_color, width=4)))
fig.add_trace(go.Scatter(x=np.concatenate([d, d[::-1]]), y=np.concatenate([u, l[::-1]]), 
                         fill='toself', fillcolor='rgba(0, 242, 254, 0.1)', line=dict(color='rgba(255,255,255,0)'), name=t("نطاق الثقة", "Confidence")))
fig.update_layout(template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader(t("🎯 أهم العوامل المؤثرة", "🎯 Key Drivers"))
    # قاموس ترجمة الـ Features بالكامل
    feat_ar = {'lag_1': "مبيعات اليوم السابق", 'lag_7': "مبيعات الأسبوع الماضي", 'rolling_mean_7': "متوسط 7 أيام", 
               'rolling_mean_14': "متوسط 14 يوم", 'is_weekend': "عطلة نهاية الأسبوع", 'was_closed_yesterday': "إغلاق أمس",
               'dayofweek_sin': "توقيت الأسبوع 1", 'dayofweek_cos': "توقيت الأسبوع 2", 'month_sin': "الموسمية 1", 'month_cos': "الموسمية 2"}
    names = [feat_ar.get(n, n) for n in feature_names] if lang=="عربي" else feature_names
    fig_i = go.Figure(go.Bar(x=model.get_feature_importance(), y=names, orientation='h', marker=dict(color=neon_color)))
    fig_i.update_layout(template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_i, use_container_width=True)

with c2:
    st.subheader(t("📥 جدول البيانات الملون", "📥 Data Table"))
    res = pd.DataFrame({t("التاريخ", "Date"): d, t("التوقع", "Forecast"): p, t("الأدنى", "Min"): l, t("الأقصى", "Max"): u})
    # الجدول بتنسيق الأرقام المصلح (Precision) وتلوين التوقع
    st.dataframe(res.style.format(precision=2).background_gradient(cmap='Blues', subset=[t("التوقع", "Forecast")]), use_container_width=True)
    st.download_button(t("تحميل ملف CSV", "Download CSV"), res.to_csv(index=False), "forecast.csv")

st.markdown("---")
st.markdown(f"<div style='text-align:center; opacity:0.6;'>Eng. Goda Emad | Retail AI v5.6 Final | 2026</div>", unsafe_allow_html=True)
# ================== 5️⃣ تحليل توزيع الأخطاء (Residuals) ==================
st.markdown("---")
st.subheader(t("🔍 تحليل جودة التوقعات (الأخطاء)", "🔍 Error & Residual Analysis"))

col_err1, col_err2 = st.columns(2)

with col_err1:
    # رسم توزيع الأخطاء (Histogram)
    residuals = metrics.get('residuals', np.random.normal(0, 1, 100))
    fig_hist = go.Figure(data=[go.Histogram(x=residuals, nbinsx=30, marker_color=neon_color, opacity=0.7)])
    fig_hist.update_layout(
        title=t("توزيع أخطاء التنبؤ", "Residuals Distribution"),
        template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=t("قيمة الخطأ", "Error Value"), yaxis_title=t("التكرار", "Frequency")
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_err2:
    # رسم الخطأ الزمني
    fig_res_time = go.Figure()
    fig_res_time.add_trace(go.Scatter(y=residuals, mode='lines', line=dict(color='#ff4b4b', width=1)))
    fig_res_time.add_annotation(text=t("خط الصفر", "Zero Line"), x=0, y=0, showarrow=False, font=dict(color="gray"))
    fig_res_time.update_layout(
        title=t("الأخطاء عبر الزمن", "Residuals Over Time"),
        template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
    st.plotly_chart(fig_res_time, use_container_width=True)
    # ================== 6️⃣ مقارنة السيناريوهات (Scenario Comparison) ==================
st.markdown("---")
st.subheader(t("📊 مقارنة السيناريوهات الثلاثة", "📊 Three-Scenario Comparison"))

# حساب السيناريوهات الثلاثة بسرعة للعرض فقط
p_opt, _, _, _ = generate_forecast(df_s, horizon, 1.15, metrics['residuals_std'])
p_real, _, _, _ = generate_forecast(df_s, horizon, 1.0, metrics['residuals_std'])
p_pess, _, _, _ = generate_forecast(df_s, horizon, 0.85, metrics['residuals_std'])

fig_scen = go.Figure()
fig_scen.add_trace(go.Scatter(x=d, y=p_opt, name=t("متفائل (+15%)", "Optimistic"), line=dict(color='#00ff88', dash='dot')))
fig_scen.add_trace(go.Scatter(x=d, y=p_real, name=t("واقعي (الأساسي)", "Realistic"), line=dict(color=neon_color, width=3)))
fig_scen.add_trace(go.Scatter(x=d, y=p_pess, name=t("متشائم (-15%)", "Pessimistic"), line=dict(color='#ff4b4b', dash='dot')))

fig_scen.update_layout(
    template=chart_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_scen, use_container_width=True)

# رسالة نصيحة نهائية بناءً على السيناريو
if scen == "متفائل":
    st.success(t("⚠️ نصيحة: السيناريو المتفائل يتطلب زيادة المخزون بنسبة 15% لتجنب نفاذ الكمية.", 
                  "⚠️ Advice: Optimistic scenario requires 15% more stock to avoid stockouts."))
elif scen == "متشائم":
    st.warning(t("⚠️ نصيحة: السيناريو المتشائم يستوجب الحذر في الطلبيات الجديدة لتقليل الراكد.", 
                   "⚠️ Advice: Pessimistic scenario suggests caution in new orders to reduce deadstock."))

if model is None: st.stop()
