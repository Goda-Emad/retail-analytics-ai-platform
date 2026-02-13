import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import time

# =========================================================
# 🔒 Forecast Guardrail
# =========================================================
def apply_forecast_guardrail(forecast_values, historical_series):
    forecast = pd.Series(forecast_values).copy()
    hist_max = historical_series.max()
    ceiling = hist_max * 1.25
    max_daily_growth = 1.15

    for i in range(1, len(forecast)):
        allowed = forecast[i-1] * max_daily_growth
        forecast[i] = min(forecast[i], allowed)

    forecast = np.minimum(forecast, ceiling)
    forecast = forecast.rolling(window=3, min_periods=1).mean()
    return forecast.values

# =========================================================
# 🚀 دالة توليد التوقعات (الدالة التي سببت الخطأ)
# =========================================================
def generate_forecast(df, horizon, multiplier, residuals_std, use_guardrail=True):
    """
    هذه الدالة هي التي يتم استدعاؤها في الجزء السادس من الداشبورد.
    تم إضافة use_guardrail=True لحل مشكلة الـ TypeError.
    """
    # ملاحظة: تأكد أن الموديل والسكيلر معرفين عالمياً أو يتم تمريرهم
    # هنا سنفترض وجودهم أو استدعاؤهم من المتغيرات المتاحة
    
    # منطق مبسط لتوليد التوقعات (Recursive Loop)
    historical_max = df['sales'].max()
    prev_value = df['sales'].iloc[-1]
    predictions = []
    
    # --- محاكة اللوب (يجب دمج منطق التوقع الحقيقي الخاص بك هنا) ---
    # هذا مجرد هيكل لضمان عمل الكود دون أخطاء
    for step in range(horizon):
        # التوقع الافتراضي (استبدله بـ model.predict الحقيقي)
        raw_pred = prev_value * 1.05 * multiplier 
        
        if use_guardrail:
            allowed_growth = prev_value * 1.15
            ceiling = historical_max * 1.25
            safe_pred = min(raw_pred, allowed_growth, ceiling)
        else:
            safe_pred = raw_pred
            
        predictions.append(safe_pred)
        prev_value = safe_pred

    final_p = pd.Series(predictions).rolling(3, min_periods=1).mean().values
    
    # إرجاع 4 قيم ليتوافق مع p, _, _, _ في الداشبورد
    return final_p, None, None, None

# =========================================================
# 📊 Backtesting (نفس كودك بدون تغيير)
# =========================================================
def run_backtesting(_df, feature_names, _scaler, _model):
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
        except:
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
        return {'mape': 0.1, 'rmse': 0, 'r2': 0, 'residuals_std': 1.0, 
                'execution_time': 0, 'data_points': len(_df), 'features_count': len(feature_names)}

    metrics_avg = pd.DataFrame(results).mean().to_dict()
    metrics_avg['residuals_std'] = np.std(all_residuals) if all_residuals else 1.0
    metrics_avg['execution_time'] = time.time() - start_time
    metrics_avg['data_points'] = len(_df)
    metrics_avg['features_count'] = len(feature_names)

    return metrics_avg
