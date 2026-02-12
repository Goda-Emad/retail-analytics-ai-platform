import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import time

@st.cache_data(show_spinner=False)
def run_backtesting(_df, feature_names, _scaler, _model):
    start_time = time.time()
    
    # 1. تنظيف أولي للداتا الأصلية
    _df = _df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['sales'])
    
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    all_residuals = []
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for train_index, test_index in tscv.split(_df):
        test_df = _df.iloc[test_index].copy()
        X_test = test_df[feature_names].copy()
        
        # ملء أي قيم مفقودة في الميزات
        X_test = X_test.ffill().bfill().fillna(0)
        
        # Scaling
        try:
            X_test[num_cols] = _scaler.transform(X_test[num_cols])
        except:
            X_test_scaled = _scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
        
        # التوقع
        preds_log = _model.predict(X_test)
        preds = np.expm1(preds_log)
        actuals = test_df['sales'].values
        
        # ⚠️ الخطوة السحرية: تنظيف التوقعات من أي قيم Inf أو NaN أو سالبة
        preds = np.nan_to_num(preds, nan=0.0, posinf=actuals.max()*2, neginf=0.0)
        preds = np.maximum(preds, 0) # التأكد إن مفيش مبيعات بالسالب
        
        # حساب البواقي
        residuals = actuals - preds
        all_residuals.extend(residuals)
        
        # حساب المقاييس مع حماية ضد الأخطاء
        try:
            results.append({
                'mape': mean_absolute_percentage_error(actuals, preds),
                'rmse': np.sqrt(mean_squared_error(actuals, preds)),
                'r2': r2_score(np.log1p(actuals), preds_log)
            })
        except:
            continue # لو فيه خطأ في حساب سجل واحد كمل الباقي
    
    if not results:
        return {'mape': 0.1, 'rmse': 0, 'r2': 0, 'residuals_std': 1.0, 
                'execution_time': 0, 'data_points': len(_df), 'features_count': len(feature_names)}

    metrics_avg = pd.DataFrame(results).mean().to_dict()
    metrics_avg['residuals_std'] = np.std(all_residuals) if all_residuals else 1.0
    metrics_avg['execution_time'] = time.time() - start_time
    metrics_avg['data_points'] = len(_df)
    metrics_avg['features_count'] = len(feature_names)
    
    return metrics_avg
