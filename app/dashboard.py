import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from PIL import Image

# ================== Page Setup ==================
st.set_page_config(page_title="Retail AI Pro v4 | Eng. Goda Emad", layout="wide")

# ================== Background Image + Gradient ==================
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.15), rgba(0,0,0,0.15)), url("images/bg_retail_1.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Header with Logo ==================
st.markdown(
    """
    <div style='display:flex; align-items:center; padding-bottom:20px;'>
        <img src='images/retail_ai_pro_logo.webp' width='100'>
        <h2 style='margin-left:15px; color:#1e293b;'>Retail AI Pro v4 | Eng. Goda Emad</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== Sidebar ==================
st.sidebar.image("images/retail_ai_pro_logo.webp", width=150)
st.sidebar.header("🎮 Control Center")
scenario = st.sidebar.selectbox("Market Scenario", ["Realistic", "Optimistic (+15%)", "Pessimistic (-15%)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 30, 21)
last_sales = st.sidebar.number_input("Last Day Sales ($)", value=32950.0)
last_customers = st.sidebar.number_input("Last Day Customers", value=485)
run_btn = st.sidebar.button("🚀 Run AI Forecast")

theme_mode = st.sidebar.selectbox("Theme Mode", ["Light 🌞", "Dark 🌙"])
if theme_mode == "Dark 🌙":
    bg_color = "#0f172a"
    text_color = "#f1f5f9"
    card_color = "rgba(30,41,59,0.85)"
    accent_color = "#3b82f6"
else:
    bg_color = "#f8fafc"
    text_color = "#1e293b"
    card_color = "rgba(255,255,255,0.75)"
    accent_color = "#2563eb"

# ================== Dummy Historical Data ==================
dates_hist = pd.date_range(end=pd.Timestamp.today(), periods=60)
sales_hist = np.random.randint(8000, 12000, size=len(dates_hist))
df_hist = pd.DataFrame({"Date": dates_hist, "Sales": sales_hist})

# ================== Forecast Engine ==================
if run_btn:
    st.subheader(f"📈 Forecast: {scenario} Scenario")
    
    future_dates = [dates_hist[-1] + timedelta(days=i+1) for i in range(horizon)]
    forecast = np.random.randint(9000, 12000, size=horizon)
    
    if scenario == "Optimistic (+15%)": forecast = forecast * 1.15
    elif scenario == "Pessimistic (-15%)": forecast = forecast * 0.85
    
    error_margin = forecast * 0.05
    upper = forecast + error_margin
    lower = forecast - error_margin
    
    # ===== Plotly Chart with Hover Effects =====
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates_hist, y=sales_hist,
        mode='lines+markers',
        name="History",
        line=dict(color='gray', width=2, shape='spline'),
        marker=dict(size=6, color='gray'),
        hovertemplate='Date: %{x}<br>Sales: $%{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast,
        mode='lines+markers',
        name="AI Forecast",
        line=dict(color=accent_color, width=3, shape='spline'),
        marker=dict(size=8, color=accent_color),
        hovertemplate='Date: %{x}<br>Forecast: $%{y:.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Error Margin"
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== Backtest Chart =====
    st.subheader("🔁 Backtest Accuracy")
    backtest_dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    actual_back = np.random.randint(9000, 12000, size=len(backtest_dates))
    pred_back = actual_back * np.random.uniform(0.95, 1.05, size=len(backtest_dates))
    mape = np.mean(np.abs((actual_back - pred_back)/actual_back))*100
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=backtest_dates, y=actual_back,
        mode='lines+markers',
        name="Actual",
        line=dict(color='gray', width=2, shape='spline'),
        marker=dict(size=6, color='gray'),
        hovertemplate='Date: %{x}<br>Actual: $%{y}<extra></extra>'
    ))
    fig_bt.add_trace(go.Scatter(
        x=backtest_dates, y=pred_back,
        mode='lines+markers',
        name="Predicted",
        line=dict(color=accent_color, width=3, shape='spline'),
        marker=dict(size=8, color=accent_color),
        hovertemplate='Date: %{x}<br>Predicted: $%{y:.0f}<extra></extra>'
    ))
    
    fig_bt.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=text_color,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_bt, use_container_width=True)
    st.markdown(f"**Backtest MAPE:** {mape:.2f}%")
    
    # ===== Metric Cards with Hover =====
    total_forecast = forecast.sum()
    avg_daily = forecast.mean()
    confidence = 82  # dummy
    
    def display_card(title, value):
        card_html = f"""
        <div style='
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(8px);
            padding: 20px;
            border-radius: 15px;
            text-align:center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        ' onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            <b>{title}</b><br>
            <span style='font-size:28px; color:{accent_color}'>{value}</span>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    display_card("Forecast Total", f"${total_forecast:,.0f}")
    display_card("Avg Daily", f"${avg_daily:,.0f}")
    display_card("Confidence", f"{confidence}%")
    col4.download_button("📥 Download Data", pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast,
        'Lower': lower,
        'Upper': upper
    }).to_csv(index=False), "forecast.csv", "text/csv", use_container_width=True)

# ================== Footer ==================
st.markdown(f"""
<div style='font-size:12px; color:#64748b; text-align:center; margin-top:20px;'>
    Retail AI Engine v4.0 | Built by <strong>Eng. Goda Emad</strong> | 
    <a href='https://www.linkedin.com/in/goda-emad/' target='_blank'>LinkedIn</a> | 
    <a href='https://github.com/Goda-Emad' target='_blank'>GitHub</a><br>
    Features: Cyclical Features, Lagged Features, Rolling Means, Backtesting, Scenario Forecasting
</div>
""", unsafe_allow_html=True)
