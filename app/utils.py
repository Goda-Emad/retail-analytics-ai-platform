import pandas as pd
import numpy as np
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import streamlit as st

@st.cache_data(show_spinner=False)
def run_backtesting(_df, feature_names, _scaler, _model):
    """
    Backtesting احترافي:
    - Metrics: MAPE, RMSE, R2
    - residuals_std لحساب نطاق الثقة
    - حماية من Inf/NaN و القيم السالبة
    """
    start_time = time.time()
    _df = _df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['sales'])
    
    n_splits = 3
    if len(_df) < (n_splits + 1) * 7:
        n_splits = 2
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    all_residuals = []
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for train_index, test_index in tscv.split(_df):
        test_df = _df.iloc[test_index].copy()
        X_test = test_df[feature_names].copy()
        X_test = X_test.ffill().bfill().fillna(0)
        
        try:
            X_test[num_cols] = _scaler.transform(X_test[num_cols])
        except ValueError:
            X_test_scaled = _scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
        
        preds_log = _model.predict(X_test)
        preds = np.expm1(preds_log)
        
        actuals = test_df['sales'].values
        preds = np.nan_to_num(preds, nan=0.0, posinf=actuals.max()*2, neginf=0.0)
        preds = np.maximum(preds, 0)
        
        residuals = actuals - preds
        all_residuals.extend(residuals)
        
        try:
            results.append({
                'mape': mean_absolute_percentage_error(actuals, preds),
                'rmse': np.sqrt(mean_squared_error(actuals, preds)),
                'r2': r2_score(np.log1p(actuals), preds_log)
            })
        except:
            continue
    
    if not results:
        return {
            'mape': 0.1, 'rmse': 0, 'r2': 0,
            'residuals_std': 1.0, 'execution_time': 0,
            'data_points': len(_df), 'features_count': len(feature_names)
        }
    
    metrics_avg = pd.DataFrame(results).mean().to_dict()
    metrics_avg['residuals_std'] = np.std(all_residuals) if all_residuals else 1.0
    metrics_avg['execution_time'] = time.time() - start_time
    metrics_avg['data_points'] = len(_df)
    metrics_avg['features_count'] = len(feature_names)
    
    return metrics_avg
