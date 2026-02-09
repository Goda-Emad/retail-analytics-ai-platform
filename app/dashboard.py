import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

# ================= 1️⃣ إعداد الصفحة =================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================= 2️⃣ CSS واجهة احترافية =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
                url('https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&w=1920&q=80');
    background-size: cover; background-attachment: fixed;
}
section[data-testid="stSidebar"] {
    background-color: rgba(10,18,29,0.95) !important;
    border-right: 5px solid #00d4ff; box-shadow:2px 0 10px rgba(0,0,0,0.3);
}
section[data-testid="stSidebar"] .stMarkdown, label, p, h2, h3, span {
    color: #f0f0f0 !important; font-family: 'Segoe UI', sans-serif;
}
.stButton > button {
    background-color: #00d4ff !important; color: white !important;
    font-weight: bold; border-radius:12px; padding:10px 20px; border:none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color:#00b0e0 !important; transform: translateY(-2px);
    box-shadow:0 4px 10px rgba(0,212,255,0.4);
}
.stPlotlyChart {
    background: rgba(255,255,255,0.95); border-radius:25px; padding:25px;
    box-shadow:0 15px 40px rgba(0,0,0,0.15); border:1px solid rgba(0,212,255,0.3);
}
.main-title {
    color: #0c151f; font-size:3.5rem; font-weight:900;
    text-align:center; text-transform:uppercase; letter-spacing:3px;
    margin-top:-60px; margin-bottom:40px; padding:10px 0;
    border-bottom:6px solid #00d4ff; display:inline-block; width:auto;
}
.metric-container {
    background: rgba(255,255,255,0.9); border-radius:20px; padding:20px;
    text-align:center; box-shadow:0 8px 20px rgba(0,0,0,0.1);
    border:1px solid rgba(0,212,255,0.2);
}
.metric-value { font-size:3.5em; font-weight:bold; color:#00d4ff; margin-bottom:5px;}
.metric-label { font-size:1.2em; color:#334155;}
</style>
""", unsafe_allow_html=True)

# ================= 3️⃣ تحميل البيانات والموديل =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model_v2.pkl")
FEAT_PATH = os.path.join(BASE_DIR, "..", "model", "feature_names.pkl")

@st.cache_data
def load_essentials():
    try:
        df = pd.read_parquet(DATA_PATH)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEAT_PATH)
        return df, model, feature_names
    except Exception as e:
        st.error(f"Error loading files. Ensure {DATA_PATH}, {MODEL_PATH}, {FEAT_PATH} exist. Error: {e}")
        st.stop()

df, model, feature_names = load_essentials()

# ================= 4️⃣ Sidebar – Inputs =================
with st.sidebar:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
    st.markdown("<h2 style='margin-bottom:0;'>Eng. Goda Emad</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#00d4ff !important; font-size:1.1em;'>AI & Retail Analytics Expert</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📊 Live Forecast Input")
    default_lag1 = float(df['total_amount'].iloc[-1]) if not df.empty else 1000.0
    default_lag7 = float(df['total_amount'].iloc[-7]) if not df.empty and len(df)>=7 else 950.0
    
    sim_lag1 = st.number_input("Yesterday's Sales ($)", min_value=0.0, max_value=float(df['total_amount'].max()*1.5), value=default_lag1, step=10.0)
    sim_lag7 = st.number_input("Last Week's Sales ($)", min_value=0.0, max_value=float(df['total_amount'].max()*1.5), value=default_lag7, step=10.0)
    forecast_days = st.slider("Forecast Days", 7, 60, 30)

    st.markdown("---")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/goda-emad/)")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Code-lightgrey?style=for-the-badge&logo=github)](https://github.com/Goda-Emad)")

# ================= 5️⃣ Main Title =================
st.markdown("<center><h1 class='main-title'>Smart Retail Forecasting AI</h1></center>", unsafe_allow_html=True)

# ================= 6️⃣ Dynamic Forecast Function =================
def generate_dynamic_forecast(model, df_hist, feature_names, start_lag1, start_lag7, days=30):
    future_preds = []
    last_date = df_hist['InvoiceDate'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    current_history = list(df_hist['total_amount'].tail(30))
    current_history[-1] = start_lag1
    
    for i in range(days):
        d = future_dates[i]
        feat_values = {
            'day': d.day,
            'month': d.month,
            'dayofweek': d.dayofweek,
            'is_weekend': 1 if d.dayofweek in [4,5] else 0,
            'rolling_mean_7': pd.Series(current_history[-7:]).mean(),
            'lag_1': current_history[-1],
            'lag_7': current_history[-7] if len(current_history)>=7 else current_history[-1]
        }
        input_data = [feat_values[f] for f in feature_names]
        pred = model.predict([input_data])[0]
        future_preds.append(pred)
        current_history.append(pred)
    return future_dates, future_preds

# ================= 7️⃣ Generate Forecast =================
f_dates, f_preds = generate_dynamic_forecast(model, df, feature_names, sim_lag1, sim_lag7, days=forecast_days)
forecast_df = pd.DataFrame({'Date': f_dates, 'Sales': f_preds})

# ================= 8️⃣ Color-coded Peaks & Lows =================
peak_idx = forecast_df['Sales'].idxmax()
low_idx = forecast_df['Sales'].idxmin()
forecast_df['Color'] = '#00d4ff'
forecast_df.loc[peak_idx, 'Color'] = '#ff4d4d'
forecast_df.loc[low_idx, 'Color'] = '#34eb77'

# ================= 9️⃣ Plotly Chart =================
fig_final = go.Figure()

# Historical sales
fig_final.add_trace(go.Scatter(
    x=df['InvoiceDate'].tail(15), y=df['total_amount'].tail(15),
    name="Actual Sales", mode='lines+markers',
    line=dict(color="#334155", width=3),
    marker=dict(size=8)
))

# Forecast
fig_final.add_trace(go.Scatter(
    x=forecast_df['Date'], y=forecast_df['Sales'],
    name="AI Forecast", mode='lines+markers',
    line=dict(color="#00d4ff", width=3, dash='dot'),
    marker=dict(color=forecast_df['Color'], size=10, symbol='star')
))

fig_final.update_layout(template='plotly_white', height=600, hovermode="x unified",
                        title_text="Historical Sales & AI Future Projections",
                        xaxis_title="Date", yaxis_title="Sales ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.plotly_chart(fig_final, use_container_width=True)

# =================  🔟 Metrics =================
st.markdown("---")
st.markdown("### ✨ Key AI Insights")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-container'><div class='metric-label'>Tomorrow's Prediction</div><div class='metric-value'>${f_preds[0]:,.2f}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-container'><div class='metric-label'>Next 7 Days Total</div><div class='metric-value'>${sum(f_preds[:7]):,.2f}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-container'><div class='metric-label'>Highest Predicted Day</div><div class='metric-value'>${max(f_preds):,.2f}</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-container'><div class='metric-label'>Lowest Predicted Day</div><div class='metric-value'>${min(f_preds):,.2f}</div></div>", unsafe_allow_html=True)

# ================= 1️⃣1️⃣ Download CSV =================
csv_buffer = BytesIO()
forecast_df.to_csv(csv_buffer, index=False)
st.download_button(label="📥 Download Forecast CSV", data=csv_buffer, file_name="forecast.csv", mime="text/csv")

# ================= 1️⃣2️⃣ Footer =================
st.markdown("---")
st.markdown(f"<center><p style='color:#334155;'>Developed by <b>Eng. Goda Emad</b> | Powered by CatBoost v2 AI | {datetime.now().year}</p></center>", unsafe_allow_html=True)
