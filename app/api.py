from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Retail AI Prediction Engine", version="2.0")

# إعداد المسارات
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "catboost_sales_model_10features.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "scaler_10features.pkl")
FEATURES_PATH = os.path.join(CURRENT_DIR, "feature_names_10features.pkl")

# تحميل الأصول عند بدء التشغيل
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.get("/")
def health_check():
    return {"status": "Running", "model": "CatBoost_10_Features", "framework": "FastAPI"}

@app.post("/predict")
async def predict_sales(data: dict):
    """
    توقع مبيعات يوم واحد بناءً على الميزات المرسلة
    """
    try:
        # تحويل البيانات لـ DataFrame
        X_df = pd.DataFrame([data])
        
        # 1. التأكد من ترتيب الأعمدة
        X_df = X_df[feature_names]
        
        # 2. عمل Scaling للأعمدة الرقمية
        num_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14']
        X_df[num_cols] = scaler.transform(X_df[num_cols])
        
        # 3. التوقع وعكس الـ Log
        log_pred = model.predict(X_df)[0]
        prediction = np.expm1(log_pred)
        
        return {
            "prediction": round(float(prediction), 2),
            "unit": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
