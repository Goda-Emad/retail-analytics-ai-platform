import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

# إضافة Cache عشان ما يتحسبش كل شوية
@st.cache_data(show_spinner=False)
def run_backtesting(_df, feature_names, _scaler, _model):
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    all_residuals = []
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for train_index, test_index in tscv.split(_df):
        test_df = _df.iloc[test_index]
        X_test = test_df[feature_names].copy()
        
        # الترتيب الصحيح قبل الـ Scaling
        X_test = X_test[feature_names]
        X_test[num_cols] = _scaler.transform(X_test[num_cols])
        
        preds_log = _model.predict(X_test)
        preds = np.expm1(preds_log)
        actuals = test_df['sales']
        
        # حساب الـ Residuals (الأخطاء) لنطاق الثقة
        residuals = actuals - preds
        all_residuals.extend(residuals)
        
        results.append({
            'mape': mean_absolute_percentage_error(actuals, preds),
            'rmse': np.sqrt(mean_squared_error(actuals, preds)),
            'r2': r2_score(np.log1p(actuals), preds_log)
        })
    
    metrics_avg = pd.DataFrame(results).mean().to_dict()
    metrics_avg['residuals_std'] = np.std(all_residuals) # الـ STD الحقيقي للأخطاء
    return metrics_avg
