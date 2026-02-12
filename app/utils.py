import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

def run_backtesting(df, feature_names, scaler, model):
    """
    وظيفة الدالة: تقييم أداء الموديل فعلياً على بياناتك التاريخية
    """
    # هنستخدم 3 تقسيمات زمنية (Folds) للتأكد من استقرار الموديل
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    
    # تحديد الأعمدة اللي محتاجة Scaling (زي ما عملت في كولاب)
    num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
    
    for train_index, test_index in tscv.split(df):
        # تقسيم البيانات لتدريب واختبار (Backtest)
        test_df = df.iloc[test_index]
        
        X_test = test_df[feature_names].copy()
        # تطبيق الـ Scaling
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        # التوقع (Log Space)
        preds_log = model.predict(X_test)
        
        # العودة للأرقام الطبيعية (expm1)
        preds = np.expm1(preds_log)
        actuals = test_df['sales']
        
        # حساب المقاييس لكل Fold
        results.append({
            'mape': mean_absolute_percentage_error(actuals, preds),
            'rmse': np.sqrt(mean_squared_error(actuals, preds)),
            'r2': r2_score(np.log1p(actuals), preds_log)
        })
    
    # حساب المتوسط النهائي لكل المقاييس
    metrics_avg = pd.DataFrame(results).mean().to_dict()
    return metrics_avg
