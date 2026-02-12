import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import time

@st.cache_data(show_spinner=False)
def run_backtesting(_df, feature_names, _scaler, _model):
    """
    دالة الـ Backtesting الاحترافية:
    - تعالج مشكلة ترتيب الأعمدة للـ Scaler.
    - تحسب دقة الموديل التاريخية.
    - توفر الـ Residuals Std لحساب نطاق الثقة (Confidence Interval).
    """
    start_time = time.time()
    
    # التأكد من وجود بيانات كافية للتقسيم (نستخدم 3 تقسيمات)
    n_splits = 3
    if len(_df) < (n_splits + 1) * 7:  # حماية لو البيانات قليلة جداً
        n_splits = 2
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    all_residuals = []
    
    # الأعمدة اللي السكيلر متدرب عليها بالترتيب (تأكد أنها مطابقة للموديل)
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for train_index, test_index in tscv.split(_df):
        test_df = _df.iloc[test_index].copy()
        
        # 1. استخراج الميزات الأساسية بالترتيب الصحيح
        X_test = test_df[feature_names].copy()
        
        # 2. معالجة القيم الناقصة (لضمان عدم حدوث خطأ في السكيلر)
        X_test = X_test.ffill().bfill() 
        
        # 3. تحويل الأعمدة الرقمية (Scaling) 
        # نستخدم القيم الموجودة في num_cols فقط كما تدرب السكيلر
        try:
            X_test[num_cols] = _scaler.transform(X_test[num_cols])
        except ValueError:
            # حل احتياطي: لو السكيلر متدرب على كل الميزات مش بس الـ 4
            X_test_scaled = _scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

        # 4. التوقع (الموديل يتوقع Log Sales)
        preds_log = _model.predict(X_test)
        preds = np.expm1(preds_log)
        actuals = test_df['sales']
        
        # 5. حساب البواقي (Residuals) لحساب نطاق الثقة لاحقاً
        residuals = actuals - preds
        all_residuals.extend(residuals)
        
        # 6. حساب مقاييس الدقة
        results.append({
            'mape': mean_absolute_percentage_error(actuals, preds),
            'rmse': np.sqrt(mean_squared_error(actuals, preds)),
            'r2': r2_score(np.log1p(actuals), preds_log)
        })
    
    # تجميع النتائج النهائية
    metrics_avg = pd.DataFrame(results).mean().to_dict()
    metrics_avg['residuals_std'] = np.std(all_residuals) if all_residuals else 0
    metrics_avg['execution_time'] = time.time() - start_time
    metrics_avg['data_points'] = len(_df)
    metrics_avg['features_count'] = len(feature_names)
    
    return metrics_avg
