import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, os, base64
from datetime import timedelta

# ================= CONFIG =================
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide")

# ================= BACKGROUND GLASS =================
def set_bg():
    base = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(base, "..", "images", "bg_retail_ai.jpg")
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{b64}") center/cover no-repeat fixed;
    }}
    .stApp::before {{
        content:"";
        position:fixed; inset:0;
        backdrop-filter: blur(16px);
        background: rgba(0,0,20,0.55);
        z-index:0;
    }}
    .block-container {{
        position:relative; z-index:1;
        background: rgba(20,30,60,0.55);
        border-radius:25px;
        padding:2rem;
        backdrop-filter: blur(22px);
    }}
    h1,h2,h3,h4,p,span,label {{color:white!important}}
    </style>
    """, unsafe_allow_html=True)

set_bg()

# ================= LOAD FILES =================
@st.cache_resource
def load_all():
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base,"catboost_sales_model_10features.pkl"))
    scaler = joblib.load(os.path.join(base,"scaler_10features.pkl"))
    features = joblib.load(os.path.join(base,"feature_names_10features.pkl"))
    data = pd.read_parquet(os.path.join(base,"daily_sales_ready_10features.parquet"))
    fi = joblib.load(os.path.join(base,"feature_importance_10features.pkl"))

    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').set_index('date')
    return model, scaler, features, data, fi

model, scaler, feature_names, df, fi_df = load_all()

# ================= SIDEBAR =================
lang = st.sidebar.selectbox("Language / اللغة", ["English","عربي"])
days = st.sidebar.slider("Forecast Days",7,60,14)
run = st.sidebar.button("🚀 Run Model / تشغيل الموديل")

# ================= KPI =================
st.title("Retail AI Forecast Dashboard" if lang=="English" else "لوحة التوقعات الذكية")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Sales", f"${df.sales.sum():,.0f}")
c2.metric("Last 30D Avg", f"${df.sales.tail(30).mean():,.0f}")
c3.metric("Best Day", f"${df.sales.max():,.0f}")
c4.metric("Data Points", len(df))

# ================= FORECAST ENGINE =================
def make_forecast(history, n):
    preds, lows, ups = [],[],[]
    cur = history.copy()

    for i in range(n):
        nd = cur.index[-1] + timedelta(days=1)

        feat = {
            'dayofweek_sin': np.sin(2*np.pi*nd.dayofweek/7),
            'dayofweek_cos': np.cos(2*np.pi*nd.dayofweek/7),
            'month_sin': np.sin(2*np.pi*(nd.month-1)/12),
            'month_cos': np.cos(2*np.pi*(nd.month-1)/12),
            'lag_1': cur.sales.iloc[-1],
            'lag_7': cur.sales.iloc[-7],
            'rolling_mean_7': cur.sales.tail(7).mean(),
            'rolling_mean_14': cur.sales.tail(14).mean(),
            'is_weekend': 1 if nd.dayofweek>=5 else 0,
            'was_closed_yesterday': 1 if cur.sales.iloc[-1]==0 else 0
        }

        X = pd.DataFrame([feat])[feature_names]
        X[['lag_1','lag_7','rolling_mean_7','rolling_mean_14']] = scaler.transform(
            X[['lag_1','lag_7','rolling_mean_7','rolling_mean_14']]
        )

        p = np.expm1(model.predict(X)[0])
        preds.append(p)
        lows.append(p*0.9)
        ups.append(p*1.1)
        cur.loc[nd]=p

    return preds, lows, ups, cur.index[-n:]

# ================= MAIN CHART =================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-60:], y=df.sales.tail(60), name="History"))

if run:
    preds,lows,ups,dates = make_forecast(df[['sales']], days)

    fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(width=4)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([ups, lows[::-1]]),
        fill='toself', name="Confidence", opacity=0.2
    ))

fig.update_layout(template="plotly_dark",
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)')

st.plotly_chart(fig, use_container_width=True)

# ================= FEATURE IMPORTANCE =================
st.subheader("Feature Importance")
fi_fig = go.Figure(go.Bar(
    x=fi_df.Importance,
    y=fi_df.Feature,
    orientation='h'
))
fi_fig.update_layout(template="plotly_dark",
                     paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fi_fig, use_container_width=True)

# ================= TABLE + DOWNLOAD =================
if run:
    res = pd.DataFrame({
        "Date":dates,
        "Forecast":preds,
        "Min":lows,
        "Max":ups
    })
    st.dataframe(res, use_container_width=True)
    st.download_button("Download CSV", res.to_csv(index=False), "forecast.csv")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
Eng. Goda Emad |
<a href='https://github.com/Goda-Emad'>GitHub</a> |
<a href='https://www.linkedin.com/in/goda-emad/'>LinkedIn</a>
""", unsafe_allow_html=True)
