import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# 1. إعداد الصفحة والـ Dark Theme
st.set_page_config(page_title="Retail AI Pro | Eng. Goda Emad", layout="wide", initial_sidebar_state="expanded")

# --- CSS لتطبيق Dark Theme واستايلات احترافية ---
st.markdown("""
    <style>
    /* Dark Theme */
    body { background-color: #0e1117; color: #fafafa; }
    .stApp { background-color: #0e1117; }
    .css-1d391kg { background-color: #1e222d; } /* Sidebar background */
    .stMarkdown, .stText, .stAlert, .stButton > button { color: #fafafa !important; }
    h1, h2, h3, h4, h5, h6 { color: #8d8f99; } /* Light grey for headers */

    /* Custom Cards for Metrics */
    .metric-card {
        background-color: #1e222d; /* Darker background for cards */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 15px;
        color: #e0e0e0;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #00d4ff; /* Accent color for values */
    }
    .metric-title {
        font-size: 1.1em;
        color: #8d8f99;
    }

    /* Plotly Charts Theme */
    .stPlotlyChart {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. الهوية الشخصية وروابط التواصل في الـ Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("### **Eng. Goda Emad**")
    st.markdown("A passionate Data Scientist & ML Engineer.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1: st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/goda-emad/) ")
    with col2: st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/Goda-Emad)")
    st.markdown("---")

st.title("🚀 Advanced Retail Sales AI Platform")
st.markdown("Welcome to your state-of-the-art sales forecasting dashboard, powered by CatBoost and enhanced features.")

# 3. تحديد المسارات الديناميكية
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "daily_sales_ready.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "catboost_sales_model.pkl")

# 4. دوال تحميل البيانات والموديل مع الـ Caching
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(path):
    df = pd.read_parquet(path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# تنفيذ التحميل
try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
    # الحصول على أسماء الفيتشرز من الموديل مباشرة
    model_features = [f'f{i}' for i in range(len(model.feature_importances_))] #Fallback if names not set
    if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
         model_features = model.feature_names_
    elif len(df.columns) > 2: # Exclude InvoiceDate and total_amount
        model_features = [col for col in df.columns if col not in ['InvoiceDate', 'total_amount']]

except Exception as e:
    st.error(f"⚠️ Error loading files or model. Please check: {e}")
    st.stop()

# 5. عرض أهمية العوامل (Feature Importance) - ملون وتفاعلي
st.subheader("📊 AI Model Feature Importance")
st.markdown("Understand which factors drive the sales predictions.")

try:
    importance = model.get_feature_importance()
    fi_df = pd.DataFrame({'Feature': model_features, 'Importance': importance}).sort_values(by='Importance', ascending=True)

    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                 title="Contribution of each feature to the forecast",
                 color='Importance', color_continuous_scale='Plasma',
                 template='plotly_dark') # Dark theme for plot
    fig_fi.update_layout(showlegend=False, height=450, title_x=0.5)
    st.plotly_chart(fig_fi, use_container_width=True)
except Exception as e:
    st.info(f"Feature importance display not available. Error: {e}")

st.divider()

# 6. قسم التوقعات التاريخية والتنبؤ المستقبلي
st.subheader("📈 Sales Trend & Future Forecast")

# فلتر لاختيار النطاق الزمني
view_range = st.slider("Historical View Range (Days)", min_value=30, max_value=365, value=90)
plot_df = df.tail(view_range).set_index('InvoiceDate')

# دالة التوقع لـ N يوم
@st.cache_data(ttl=3600)
def generate_future_forecast(_model, _historical_df, days_ahead):
    last_date = _historical_df.index.max()
    future_dates = pd.date_range(start=last_date, periods=days_ahead + 1, freq='D')[1:]
    
    forecast_df = pd.DataFrame(index=future_dates)
    
    # تحضير الـ Lags والـ Rolling من البيانات التاريخية
    temp_df = _historical_df.copy()
    
    predictions = []
    
    for i in range(days_ahead):
        current_date = future_dates[i]
        
        # Build features for current_date
        current_features = {}
        current_features['day'] = current_date.day
        current_features['month'] = current_date.month
        current_features['year'] = current_date.year
        current_features['dayofweek'] = current_date.dayofweek
        current_features['dayofyear'] = current_date.dayofyear
        current_features['weekofyear'] = current_date.isocalendar().week.astype(int)
        current_features['quarter'] = current_date.quarter
        current_features['is_holiday'] = 0 # Default, or add specific holiday logic
        current_features['is_weekend'] = ((current_date.dayofweek == 5) | (current_date.dayofweek == 6)).astype(int)
        
        # Rolling means/std - needs previous 7/30 days from temp_df
        current_features['rolling_mean_7'] = temp_df['total_amount'].rolling(window=7).mean().iloc[-1]
        current_features['rolling_mean_30'] = temp_df['total_amount'].rolling(window=30).mean().iloc[-1]
        current_features['rolling_std_7'] = temp_df['total_amount'].rolling(window=7).std().iloc[-1]
        
        # Lag Features - needs previous values from temp_df
        for lag in [1, 2, 3, 7, 14, 30, 60]:
            current_features[f'lag_{lag}'] = temp_df['total_amount'].shift(lag).iloc[-1]
            
        # Ensure feature order matches model's training order
        feature_values_for_pred = [current_features[f] for f in model_features]

        pred = _model.predict([feature_values_for_pred])[0]
        predictions.append(pred)
        
        # Add prediction to temp_df to calculate next rolling/lags
        temp_df.loc[current_date, 'total_amount'] = pred
    
    forecast_df['total_amount'] = predictions
    return forecast_df

days_to_forecast = st.slider("Forecast Days Ahead", min_value=7, max_value=60, value=30)

with st.spinner(f"Generating {days_to_forecast}-day forecast..."):
    future_forecast_df = generate_future_forecast(model, plot_df.copy(), days_to_forecast)

# دمج البيانات التاريخية مع التوقعات لرسم بياني واحد
combined_df = pd.concat([plot_df['total_amount'], future_forecast_df['total_amount']], axis=0)
combined_df = combined_df.rename("Sales")

fig_combined = px.line(combined_df.reset_index(), x='index', y='Sales',
                       title=f"Historical Sales and {days_to_forecast}-Day AI Forecast",
                       labels={'index': 'Date', 'Sales': 'Sales ($)'},
                       template='plotly_dark')

# تمييز خط التوقع بلون مختلف
fig_combined.add_trace(go.Scatter(x=future_forecast_df.index, y=future_forecast_df['total_amount'],
                                   mode='lines', name='AI Forecast',
                                   line=dict(color='#ff6347', width=3, dash='dot'))) # لون أحمر منقط

fig_combined.update_traces(line=dict(color='#00d4ff', width=2), selector=dict(name='Sales'))
fig_combined.update_layout(hovermode="x unified", title_x=0.5)
st.plotly_chart(fig_combined, use_container_width=True)


# 7. قسم التوقع اليدوي (Interactive Prediction Lab)
st.sidebar.header("🧪 Prediction Lab (Adjust & Predict)")
st.sidebar.markdown("Test the AI by adjusting key parameters.")

with st.sidebar.expander("Adjust Input Features", expanded=True):
    # نأخذ قيم الفيتشرز الأكثر أهمية من أهمية العوامل
    # هذه القيمة يجب أن تعكس "أهم" قيم الـ features_names من الـ FI_DF
    # هنا تم افتراض أهمية Lag1, Lag7, Month, Day
    manual_day = st.number_input("Target Day (1-31)", value=int(pd.Timestamp.now().day), min_value=1, max_value=31)
    manual_month = st.number_input("Target Month (1-12)", value=int(pd.Timestamp.now().month), min_value=1, max_value=12)
    manual_lag1 = st.number_input("Prev Day Sales (Lag 1)", value=float(df['total_amount'].iloc[-1]), format="%.2f")
    manual_lag7 = st.number_input("Prev Week Sales (Lag 7)", value=float(df['total_amount'].iloc[-7]), format="%.2f")
    # إضافة rolling mean كمثال
    manual_rmean7 = st.number_input("Rolling Mean (Last 7 Days)", value=float(df['rolling_mean_7'].iloc[-1]), format="%.2f")

if st.sidebar.button("✨ Get Instant AI Prediction"):
    current_year = pd.Timestamp.now().year
    current_dayofweek = pd.Timestamp.now().dayofweek
    current_dayofyear = pd.Timestamp.now().dayofyear
    current_weekofyear = pd.Timestamp.now().isocalendar().week.astype(int)
    current_quarter = pd.Timestamp.now().quarter

    # بناء قائمة الفيتشرز للتوقع اليدوي، مع التأكد من الترتيب والطول
    # يجب أن تتطابق هذه القائمة مع 'model_features' بالترتيب
    manual_test_features = {
        'day': manual_day,
        'month': manual_month,
        'year': current_year,
        'dayofweek': current_dayofweek, # يمكن جعلها تفاعلية أيضاً
        'dayofyear': current_dayofyear,
        'weekofyear': current_weekofyear,
        'quarter': current_quarter,
        'is_holiday': 0, # Default to no holiday
        'is_weekend': 0, # Default to not weekend
        'rolling_mean_7': manual_rmean7,
        'rolling_mean_30': float(df['rolling_mean_30'].iloc[-1]), # يمكن جعلها تفاعلية
        'rolling_std_7': float(df['rolling_std_7'].iloc[-1]),     # يمكن جعلها تفاعلية
        'lag_1': manual_lag1,
        'lag_2': float(df['lag_2'].iloc[-1]), # يمكن جعلها تفاعلية
        'lag_3': float(df['lag_3'].iloc[-1]), # يمكن جعلها تفاعلية
        'lag_7': manual_lag7,
        'lag_14': float(df['lag_14'].iloc[-1]), # يمكن جعلها تفاعلية
        'lag_30': float(df['lag_30'].iloc[-1]), # يمكن جعلها تفاعلية
        'lag_60': float(df['lag_60'].iloc[-1])  # يمكن جعلها تفاعلية
    }
    
    # ضمان أن ترتيب وعدد الفيتشرز يطابق توقعات الموديل بالضبط
    final_manual_features_list = [manual_test_features[f] for f in model_features]
    
    manual_prediction = model.predict([final_manual_features_list])[0]
    st.sidebar.metric("🔮 AI Predicted Sales", f"${manual_prediction:,.2f}", delta="Instant Result")
    st.sidebar.success("Prediction Generated!")
    st.sidebar.balloons()

st.markdown("---")
st.caption(f"Developed with ❤️ by Eng. Goda Emad | Retail Analytics v3.0 | {pd.Timestamp.now().year}")
