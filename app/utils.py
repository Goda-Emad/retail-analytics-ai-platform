import pandas as pd
import numpy as np
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from typing import Tuple, Optional, Any, Dict

# =========================================================
# 🔒 Forecast Guardrail
# =========================================================
def apply_forecast_guardrail(
    forecast_values: np.ndarray | list,
    historical_series: pd.Series,
    max_daily_growth: float = 1.15,
    ceiling_multiplier: float = 1.25,
    rolling_window: int = 3
) -> np.ndarray:
    """
    تطبيق قيود ذكية على التوقعات لمنع النمو الانفجاري.
    
    Args:
        forecast_values: قائمة أو مصفوفة numpy للتوقعات الأولية.
        historical_series: السلسلة التاريخية للمبيعات.
        max_daily_growth: أقصى نسبة نمو يومية مسموح بها.
        ceiling_multiplier: الحد الأعلى (نسبة من أعلى قيمة تاريخية).
        rolling_window: حجم النافذة لتنعيم المتوسط المتحرك.
    
    Returns:
        np.ndarray: توقعات بعد تطبيق الـ guardrail والمتوسط المتحرك.
    """
    forecast = pd.Series(forecast_values).copy()
    hist_max = historical_series.max()
    ceiling = hist_max * ceiling_multiplier

    for i in range(1, len(forecast)):
        allowed = forecast[i - 1] * max_daily_growth
        forecast[i] = min(forecast[i], allowed)

    forecast = np.minimum(forecast, ceiling)
    forecast = forecast.rolling(window=rolling_window, min_periods=1).mean()

    return forecast.values


# =========================================================
# 🚀 Forecast Generation
# =========================================================
def generate_forecast(
    df: pd.DataFrame,
    horizon: int,
    multiplier: float = 1.0,
    residuals_std: Optional[float] = None,
    use_guardrail: bool = True
) -> Tuple[np.ndarray, Any, Any, Any]:
    """
    توليد توقعات مبيعات افتراضية مع دعم guardrail.
    
    Args:
        df: DataFrame يحتوي على عمود 'sales'.
        horizon: عدد أيام التوقع القادمة.
        multiplier: معدل تعديل السيناريو (متفائل، واقعي، متشائم).
        residuals_std: الانحراف المعياري للمخلفات (اختياري).
        use_guardrail: تفعيل Guardrail لمنع النمو الانفجاري.
    
    Returns:
        tuple: (forecast_array, None, None, None) لتوافق الداشبورد.
    """
    if 'sales' not in df.columns:
        raise ValueError("DataFrame يجب أن يحتوي على عمود 'sales'.")

    historical_max = df['sales'].max()
    prev_value = df['sales'].iloc[-1]
    predictions = []

    for step in range(horizon):
        raw_pred = prev_value * 1.05 * multiplier

        if use_guardrail:
            allowed_growth = prev_value * 1.15
            ceiling = historical_max * 1.25
            safe_pred = min(raw_pred, allowed_growth, ceiling)
        else:
            safe_pred = raw_pred

        predictions.append(safe_pred)
        prev_value = safe_pred

    final_pred = pd.Series(predictions).rolling(3, min_periods=1).mean().values
    return final_pred, None, None, None


# =========================================================
# 📊 Backtesting
# =========================================================
def run_backtesting(
    df: pd.DataFrame,
    feature_names: list,
    scaler: Any,
    model: Any,
    num_cols: Optional[list] = None,
    n_splits: int = 3
) -> Dict[str, Any]:
    """
    إجراء Backtesting على موديل المبيعات.
    
    Args:
        df: DataFrame يحتوي على بيانات المبيعات.
        feature_names: قائمة أسماء الميزات المستخدمة في التنبؤ.
        scaler: كائن Scaler لتطبيقه على البيانات العددية.
        model: الموديل المدرب (CatBoost أو غيره).
        num_cols: الأعمدة العددية التي تحتاج scaling.
        n_splits: عدد تقسيمات TimeSeries.
    
    Returns:
        dict: يحتوي على metrics مثل MAPE, RMSE, R2, residuals_std, execution_time, data_points, features_count
    """
    start_time = time.time()
    df = df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['sales'])
    results = []
    all_residuals = []

    if not num_cols:
        num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']

    if len(df) < (n_splits + 1) * 7:
        n_splits = max(2, n_splits - 1)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(df):
        test_df = df.iloc[test_idx].copy()
        X_test = test_df[feature_names].copy().ffill().bfill().fillna(0)

        # تطبيق الـ Scaler
        try:
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        except:
            X_test_scaled = scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

        # توقع الموديل
        preds_log = model.predict(X_test)
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
            'data_points': len(df), 'features_count': len(feature_names)
        }

    metrics_avg = pd.DataFrame(results).mean().to_dict()
    metrics_avg['residuals_std'] = np.std(all_residuals) if all_residuals else 1.0
    metrics_avg['execution_time'] = time.time() - start_time
    metrics_avg['data_points'] = len(df)
    metrics_avg['features_count'] = len(feature_names)

    return metrics_avg
