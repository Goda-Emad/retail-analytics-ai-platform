# ================== app.py ==================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import base64
import requests
import os

# ================== Paths ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DATA_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
PRODUCT_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ================== Load Model & Data ==================
@st.cache_resource
def load_data():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    
    df_sales = pd.read_parquet(DATA_PATH)
    df_sales = df_sales.reset_index()
    df_sales[df_sales.columns[0]] = pd.to_datetime(df_sales[df_sales.columns[0]])
    df_sales = df_sales.set_index(df_sales.columns[0])
    
    df_products = pd.read_parquet(PRODUCT_PATH)
    if 'Description' in df_products.columns:
        df_products = df_products.rename(columns={'Description':'Product'})
    
    return model, feature_names, df_sales, df_products

model, feature_names, sales_df, product_df = load_data()

# ================== Background ==================
BG_URL = "https://images.unsplash.com/photo-1598032891587-73bb75889144?auto=format&fit=crop&w=1950&q=80"
bg_base64 = base64.b64encode(requests.get(BG_URL).content).decode()

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: fixed; top:0; left:0; width:100%; height:100%;
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    z-index: -1;
}}
.header-container {{
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    padding: 25px; border-radius: 20px; text-align:center;
    border: 1px solid rgba(255,255,255,0.15);
}}
.metric-box {{
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    padding:25px; border-radius:15px; text-align:center;
    border:1px solid rgba(255,255,255,0.15);
}}
.metric-box h2 {{
    color:#00D4FF !important;
    margin: 10px 0 0 0;
}}
</style>
""", unsafe_allow_html=True)

# ================== Forecast Functions ==================
def prepare_features(current_hist, next_date):
    day_sin = np.sin(2*np.pi*next_date.dayofweek/7)
    day_cos = np.cos(2*np.pi*next_date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(next_date.isocalendar().week % 52)/52)
    week_cos = np.cos(2*np.pi*(next_date.isocalendar().week % 52)/52)
    month_sin = np.sin(2*np.pi*next_date.month/12)
    month_cos = np.cos(2*np.pi*next_date.month/12)
    
    features_dict = {
        'day_sin': day_sin,'day_cos':day_cos,
        'week_sin':week_sin,'week_cos':week_cos,
        'month_sin':month_sin,'month_cos':month_cos,
        'is_month_end': int(next_date.is_month_end),
        'lag_1': current_hist.iloc[-1],
        'lag_7': current_hist.iloc[-7] if len(current_hist)>=7 else current_hist.mean(),
        'lag_30': current_hist.iloc[-30] if len(current_hist)>=30 else current_hist.mean(),
        'rolling_mean_7': current_hist[-7:].mean() if len(current_hist)>=7 else current_hist.mean(),
        'rolling_mean_30': current_hist[-30:].mean() if len(current_hist)>=30 else current_hist.mean()
    }
    X_df = pd.DataFrame([features_dict])
    for feat in feature_names:
        if feat not in X_df.columns:
            X_df[feat] = 0
    X_df = X_df[feature_names].astype(float)
    return X_df

def generate_forecast(hist_series, horizon, scenario, noise_val):
    forecast_values = []
    current_hist = hist_series.copy()
    with st.spinner("🔮 التنبؤ جاري..."):
        for i in range(horizon):
            next_date = current_hist.index[-1] + timedelta(days=1)
            X_df = prepare_features(current_hist, next_date)
            pred = model.predict(X_df)[0]
            if scenario=="متفائل (+15%)": pred*=1.15
            elif scenario=="متشائم (-15%)": pred*=0.85
            pred = max(0, pred*(1+np.random.normal(0, noise_val)))
            forecast_values.append(pred)
            current_hist.loc[next_date] = pred
    st.success(f"✅ التنبؤ تحت سيناريو '{scenario}' اكتمل!")
    return np.array(forecast_values), current_hist.index[-horizon:]

# ================== Sidebar ==================
st.sidebar.header("🛒 Control Panel")
scenario = st.sidebar.selectbox("اختر سيناريو السوق", ["واقعي","متفائل (+15%)","متشائم (-15%)"])
horizon = st.sidebar.slider("عدد أيام التوقع",7,30,14)
noise_val = st.sidebar.slider("تقلب السوق",0.0,0.1,0.03)
product_options = ["All Products"] + product_df['Product'].tolist()
product_filter = st.sidebar.selectbox("اختار منتج (اختياري)", product_options)
run_btn = st.sidebar.button("🚀 تشغيل التوقعات")

# ================== Header ==================
st.markdown(f"<div class='header-container'><h1 style='color:#00D4FF'>Retail AI Pro</h1><p>Eng. Goda Emad | Intelligent Supermarket Forecasting</p></div>", unsafe_allow_html=True)

# ================== Main ==================
if run_btn:
    df_hist = sales_df.copy()
    if product_filter!="All Products":
        if product_filter in product_df['Product'].values:
            df_hist = df_hist[df_hist['Product']==product_filter]

    preds, dates = generate_forecast(df_hist['Daily_Sales'], horizon, scenario, noise_val)
    
    # ==== Line Chart ====
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Daily_Sales'], name="البيانات التاريخية", line=dict(color="gray")))
    fig1.add_trace(go.Scatter(x=dates, y=preds, name=f"توقع ({scenario})", line=dict(color="cyan", width=3)))
    st.plotly_chart(fig1,use_container_width=True)
    
    # ==== Top Products Bar Chart ====
    top_products = product_df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig2 = go.Figure([go.Bar(x=top_products.index, y=top_products.values, marker_color='orange')])
    fig2.update_layout(title="أفضل 10 منتجات", xaxis_title="المنتج", yaxis_title="الكمية")
    st.plotly_chart(fig2,use_container_width=True)
    
    # ==== Product Share Pie Chart ====
    product_share = product_df.groupby('Product')['Quantity'].sum()
    fig3 = go.Figure([go.Pie(labels=product_share.index, values=product_share.values)])
    fig3.update_layout(title="حصة المنتجات من المبيعات")
    st.plotly_chart(fig3,use_container_width=True)
    
    # ==== KPI Cards ====
    total_sales = preds.sum()
    avg_sales = total_sales / horizon
    c1,c2 = st.columns(2)
    c1.metric("إجمالي المبيعات المتوقعة",f"${total_sales:,.0f}")
    c2.metric("المعدل اليومي المتوقع",f"${avg_sales:,.0f}")
    
    # ==== Download Button ====
    df_download = pd.DataFrame({'Date':dates,'Forecast':preds})
    st.download_button("⬇️ تحميل التنبؤ كـ CSV", df_download.to_csv(index=False), "forecast.csv","text/csv")
    
else:
    st.info("اختر المعايير في الشريط الجانبي واضغط 🚀 تشغيل التوقعات")
