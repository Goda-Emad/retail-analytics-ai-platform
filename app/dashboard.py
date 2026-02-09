import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail AI Forecast Dashboard", layout="wide")

st.title("📈 Retail Sales Forecasting AI Dashboard")

# ================== Load Data & Model ==================
@st.cache_data
def load_data():
    df = pd.read_csv("data/daily_sales_ready.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load("model/catboost_sales_model.pkl")
    return model

df = load_data()
model = load_model()

# ================== Show Historical Sales ==================
st.subheader("Historical Daily Sales")

daily = df[['InvoiceDate', 'daily_sales']]
daily = daily.sort_values('InvoiceDate')

st.line_chart(daily.set_index('InvoiceDate'))

# ================== Forecast Next Days ==================
st.subheader("🔮 Forecast Next 30 Days")

last_date = daily['InvoiceDate'].max()
future_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]

# تجهيز الداتا للتوقع
future_df = pd.DataFrame()
future_df['InvoiceDate'] = future_dates
future_df['day'] = future_df['InvoiceDate'].dt.day
future_df['month'] = future_df['InvoiceDate'].dt.month
future_df['year'] = future_df['InvoiceDate'].dt.year
future_df['dayofweek'] = future_df['InvoiceDate'].dt.dayofweek

# Lag Features من آخر قيم حقيقية
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

# ================== Plot Forecast ==================
st.line_chart(
    pd.concat([
        daily.set_index('InvoiceDate'),
        future_df[['InvoiceDate','Predicted_Sales']].set_index('InvoiceDate')
    ], axis=1)
)

st.success("✅ AI Forecast Generated Successfully")

