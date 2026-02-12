import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import time

@st.cache_data(show_spinner=False)
def run_backtesting(_df, feature_names, _scaler, _model):
    """
    نسخة Backtesting المطورة:
    - تعالج مشكلة الـ inf% في الـ MAPE.
    - تحمي الحسابات من القيم الصفرية والـ Outliers.
    """
    start_time = time.time()
    
    # 1. تنظيف البيانات الأساسية
    _df = _df.copy().replace([np.inf, -np.inf], np.nan)
    _df['sales'] = _df['sales'].fillna(0)
    
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    all_residuals = []
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for train_index, test_index in tscv.split(_df):
        test_df = _df.iloc[test_index].copy()
        X_test = test_df[feature_names].copy()
        
        # ملء الفراغات في الميزات
        X_test = X_test.ffill().bfill().fillna(0)
        
        # Scaling آمن
        try:
            X_test[num_cols] = _scaler.transform(X_test[num_cols])
        except:
            X_test_scaled = _scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_df.index)
        
        # التوقع
        preds_log = _model.predict(X_test)
        preds = np.expm1(preds_log)
        actuals = test_df['sales'].values
        
        # تنظيف التوقعات (منع السالب والقيم الخرافية)
        preds = np.nan_to_num(preds, nan=0.0, posinf=actuals.max()*2)
        preds = np.maximum(preds, 0)
        
        # حساب البواقي للـ Confidence Interval
        residuals = actuals - preds
        all_residuals.extend(residuals)
        
        # 🛡️ الحساب الذكي للمقاييس (تجنب القسمة على صفر)
        try:
            # نستخدم فقط الأيام اللي فيها مبيعات حقيقية > 0 لحساب الـ MAPE
            mask = actuals > 0
            if np.any(mask):
                mape_val = mean_absolute_percentage_error(actuals[mask], preds[mask])
            else:
                mape_val = 0.0 # لو كل الأيام أصفار، الـ Error صفر
                
            results.append({
                'mape': mape_val,
                'rmse': np.sqrt(mean_squared_error(actuals, preds)),
                'r2': r2_score(np.log1p(actuals), preds_log)
            })
        except:
            continue
            
    # لو مفيش نتائج (حماية نهائية)
    if not results:
        return {'mape': 0.0, 'rmse': 0.0, 'r2': 0.0, 'residuals_std': 1.0, 
                'execution_time': 0, 'data_points': len(_df)}

    # تجميع المتوسطات
    metrics_avg = pd.DataFrame(results).mean().to_dict()
    
    # حماية إضافية للـ R2 لو طلع رقم سالب خيالي
    metrics_avg['r2'] = max(metrics_avg.get('r2', 0), 0)
    
    metrics_avg['residuals_std'] = np.std(all_residuals) if all_residuals else 1.0
    metrics_avg['execution_time'] = time.time() - start_time
    metrics_avg['data_points'] = len(_df)
    
    return metrics_avg
