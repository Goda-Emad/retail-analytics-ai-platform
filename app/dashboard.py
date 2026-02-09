import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail AI Forecast Dashboard", layout="wide")

st.title("📈 Retail Sales Forecasting AI Dashboard")

# ================== Load Data ==================
@st.cache_data
def load_data():
    df = pd.read_csv("../data/daily_sales_ready.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# ================== Load Model ==================
@st.cache_resource
def load_model():
    return joblib.load("../model/catboost_sales_model.pkl")

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ================== Historical Sales ==================
st.subheader("📊 Historical Daily Sales")

daily = df[['InvoiceDate', 'daily_sales']].sort_values('InvoiceDate')
daily = daily.set_index('InvoiceDate')

st.line_chart(daily)

# ================== Forecast Section ==================
st.subheader("🔮 Forecast Next 30 Days")

last_date = daily.index.max()
future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]

future_df = pd.DataFrame({'InvoiceDate': future_dates})
future_df['day'] = future_df['InvoiceDate'].dt.day
future_df['month'] = future_df['InvoiceDate'].dt.month
future_df['year'] = future_df['InvoiceDate'].dt.year
future_df['dayofweek'] = future_df['InvoiceDate'].dt.dayofweek

last_values = list(daily['daily_sales'].tail(7))
predictions = []

for i in range(len(future_df)):
    row = future_df.iloc[i].copy()

    row['lag_1'] = last_values[-1]
    row['lag_2'] = last_values[-2]
    row['lag_3'] = last_values[-3]
    row['lag_7'] = last_values[-7]

    features = row[['day','month','year','dayofweek','lag_1','lag_2','lag_3','lag_7']].values.reshape(1, -1)
    
    pred = model.predict(features)[0]
    predictions.append(pred)
    last_values.append(pred)

future_df['Predicted_Sales'] = predictions
future_df = future_df.set_index('InvoiceDate')

# ================== Plot Forecast ==================
st.subheader("📈 Sales Forecast vs Historical")

combined = pd.concat([
    daily.rename(columns={'daily_sales': 'Historical Sales'}),
    future_df.rename(columns={'Predicted_Sales': 'Forecasted Sales'})
], axis=1)

st.line_chart(combined)

st.success("✅ Forecast generated successfully using CatBoost AI Model")
