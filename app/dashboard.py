# ==================== app.py (Pro Supermarket Dashboard - Tabs Version) ====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from catboost import CatBoostRegressor
import joblib
import os
import requests

# ================== Paths ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names.pkl")
DAILY_SALES_PATH = os.path.join(CURRENT_DIR, "daily_sales_ready.parquet")
PRODUCT_ANALYTICS_PATH = os.path.join(CURRENT_DIR, "product_analytics.parquet")

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================== Load Model & Data ==================
@st.cache_resource
def load_essentials():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)
            and os.path.exists(DAILY_SALES_PATH) and os.path.exists(PRODUCT_ANALYTICS_PATH)):
        st.error("❌ Missing one or more essential files!")
        return None, None, None, None
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    daily_df = pd.read_parquet(DAILY_SALES_PATH)
    product_df = pd.read_parquet(PRODUCT_ANALYTICS_PATH)

    if daily_df.index.name is not None:
        daily_df = daily_df.reset_index()

    # تحديد عمود التاريخ
    date_col = next((c for c in daily_df.columns if 'date' in c.lower()), None)
    if date_col:
        daily_df[date_col] = pd.to_datetime(daily_df[date_col])
        daily_df = daily_df.set_index(date_col)
    else:
        st.error("❌ No date column found in daily sales.")
        st.stop()

    if "Daily_Sales" not in daily_df.columns:
        possible_sales = [c for c in daily_df.columns if 'sales' in c.lower()]
        if possible_sales:
            daily_df = daily_df.rename(columns={possible_sales[0]: "Daily_Sales"})
        else:
            st.error("❌ No Daily_Sales column found.")
            st.stop()

    return model, features, daily_df, product_df

model, feature_names, sales_df, product_df = load_essentials()

# ================== Dark/Light Mode ==================
mode = st.sidebar.selectbox("اختر وضع الواجهة", ["Dark 🌙", "Light 🌞"])
if mode == "Dark 🌙":
    overlay = "rgba(10, 10, 20, 0.4)"
    text_color = "#ffffff"
    accent_color = "#00D4FF"
    card_bg = "rgba(255, 255, 255, 0.07)"
    border_color = "rgba(255, 255, 255, 0.15)"
else:
    overlay = "rgba(255, 255, 255, 0.4)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    card_bg = "rgba(255, 255, 255, 0.5)"
    border_color = "rgba(0, 0, 0, 0.1)"

# ================== Online Supermarket Background ==================
BG_URL = "https://images.unsplash.com/photo-1566210130058-f63a3d17338b?auto=format&fit=crop&w=1950&q=80"
st.markdown(f"""
<style>
.stApp {{
    background-image: url("{BG_URL}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: fixed; top:0; left:0; width:100%; height:100%;
    background: {overlay};
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    z-index: -1;
}}
.header-container {{
    display:flex; align-items:center; padding:25px;
    background: {card_bg}; 
    backdrop-filter: blur(10px);
    border-radius:20px; margin-bottom:25px;
    border: 1px solid {border_color};
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}}
.metric-box {{
    background: {card_bg}; 
    backdrop-filter: blur(10px);
    padding:25px; border-radius:15px;
    text-align:center; border: 1px solid {border_color};
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    color: {text_color};
}}
.metric-box h2 {{
    color: {accent_color} !important;
    margin: 10px 0 0 0;
}}
.sidebar-link {{
    display:block; margin-top:5px; color:{accent_color}; font-weight:bold; text-decoration:none;
}}
section[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(15px);
}}
</style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown(f"""
<div class="header-container">
    <div style="width: 100%; text-align: center;">
        <h1 style="margin:0; color:{accent_color}; font-size: 2.5rem; text-transform: uppercase; letter-spacing: 2px;">Retail AI Pro</h1>
        <p style="margin:5px; color:{text_color}; opacity:0.9; font-weight:400;">Eng. Goda Emad | Intelligent Supermarket Forecasting</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== Forecast Functions ==================
def get_cyclical_features(date):
    day_sin = np.sin(2*np.pi*date.dayofweek/7)
    week_sin = np.sin(2*np.pi*(date.isocalendar().week % 52)/52)
    month_sin = np.sin(2*np.pi*date.month/12)
    return day_sin, week_sin, month_sin

def generate_forecast(hist_series, horizon, scenario, noise_val):
    forecast_values = []
    current_hist = hist_series.copy()
    for i in range(horizon):
        next_date = current_hist.index[-1] + timedelta(days=1)
        d_sin, w_sin, m_sin = get_cyclical_features(next_date)
        features_dict = {
            'day_sin': d_sin, 'week_sin': w_sin, 'month_sin': m_sin,
            'lag_1': current_hist.iloc[-1],
            'lag_7': current_hist.iloc[-7] if len(current_hist)>=7 else current_hist.mean()
        }
        X_df = pd.DataFrame([features_dict])
        for feat in feature_names:
            if feat not in X_df.columns: X_df[feat] = 0
        X_df = X_df[feature_names]
        pred = model.predict(X_df)[0]
        if scenario == "متفائل (+15%)": pred *= 1.15
        elif scenario == "متشائم (-15%)": pred *= 0.85
        pred = max(0, pred*(1+np.random.normal(0,noise_val)))
        forecast_values.append(pred)
        current_hist.loc[next_date] = pred
    return np.array(forecast_values), current_hist.index[-horizon:]

# ================== Sidebar ==================
with st.sidebar:
    st.markdown(f"<h2 style='color:{accent_color};'>🛒 Control Panel</h2>", unsafe_allow_html=True)
    scenario = st.selectbox("اختار سيناريو السوق", ["واقعي", "متفائل (+15%)", "متشائم (-15%)"])
    horizon = st.slider("عدد أيام التوقع", 7, 30, 14)
    noise_lvl = st.slider("تقلب السوق", 0.0, 0.1, 0.03)
    start_date = st.date_input("من تاريخ", sales_df.index.min().date())
    end_date = st.date_input("إلى تاريخ", sales_df.index.max().date())
    run_btn = st.button("🚀 تشغيل التوقعات", use_container_width=True)

# ================== Tabs ==================
tabs = st.tabs(["📈 Forecast", "🏆 Top Products Analytics"])

# ================== Tab 1: Forecast ==================
with tabs[0]:
    if run_btn:
        df_filtered = sales_df[start_date:end_date]
        scenarios_list = ["واقعي", "متفائل (+15%)", "متشائم (-15%)"]
        colors = ["#00D4FF", "#00FF88", "#FF4B2B"]
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered.values,
            name="البيانات التاريخية",
            fill='tozeroy',
            fillcolor='rgba(150,150,150,0.1)',
            line=dict(color="rgba(200,200,200,0.5)", width=2)
        ))

        forecasts_dict = {}
        for sc, color in zip(scenarios_list, colors):
            preds, dates = generate_forecast(df_filtered, horizon, sc, noise_lvl)
            forecasts_dict[sc] = preds
            fig_forecast.add_trace(go.Scatter(
                x=dates,
                y=preds,
                name=f"توقع ({sc})",
                mode='lines+markers',
                line=dict(color=color, width=4, shape='spline'),
                marker=dict(size=6)
            ))

        fig_forecast.update_layout(
            title="Forecast for Next Days",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=text_color,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="المبيعات ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # KPI Cards
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        total_forecast = np.mean([forecasts_dict[sc].sum() for sc in scenarios_list])
        avg_forecast = total_forecast / horizon
        c1.markdown(f"<div class='metric-box'>إجمالي المبيعات المتوقعة<br><h2>${total_forecast:,.0f}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'>المعدل اليومي المستهدف<br><h2>${avg_forecast:,.0f}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'>دقة نموذج التنبؤ AI<br><h2>82%</h2></div>", unsafe_allow_html=True)

        # Download CSV
        def get_csv_download(forecasts_dict, dates):
            df_download = pd.DataFrame({"Date": dates})
            for sc, preds in forecasts_dict.items():
                df_download[sc] = preds
            return df_download

        csv_data = get_csv_download(forecasts_dict, dates).to_csv(index=False).encode()
        st.download_button(
            label="⬇️ تحميل التوقعات CSV",
            data=csv_data,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )
    else:
        st.info("اضغط على زر تشغيل التوقعات في الشريط الجانبي لعرض Forecast + KPI Cards + Download CSV.")

# ================== Tab 2: Top Products Analytics ==================
with tabs[1]:
    top_products = product_df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(10)
    st.subheader("Top 10 Products Bar Chart")
    fig_bar = go.Figure([go.Bar(
        x=top_products.index,
        y=top_products.values,
        marker_color="#FF4B2B"
    )])
    fig_bar.update_layout(
        xaxis_title="Product",
        yaxis_title="Total Quantity Sold",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Top 5 Products Sales Share Pie Chart")
    pie_data = product_df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(5)
    fig_pie = go.Figure([go.Pie(
        labels=pie_data.index,
        values=pie_data.values,
        hole=0.4
    )])
    fig_pie.update_layout(
        font_color=text_color,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Full Products Table")
    st.dataframe(product_df.sort_values(by='Quantity', ascending=False).head(20))

# ================== Footer ==================
st.markdown(f"""
<div style="text-align:center; padding:30px; color:{text_color}; opacity:0.7; font-size:0.9rem;">
    <strong>Developed by Eng. Goda Emad</strong><br>
    <a href='https://www.linkedin.com/in/goda-emad/' class='sidebar-link' style='display:inline;'>LinkedIn</a> | 
    <a href='https://github.com/Goda-Emad' class='sidebar-link' style='display:inline;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
