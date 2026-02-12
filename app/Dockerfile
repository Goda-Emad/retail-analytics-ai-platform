# 1. استخدام نسخة خفيفة من بايثون
FROM python:3.9-slim

# 2. تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# 3. تثبيت أدوات النظام الضرورية
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. نسخ ملف المتطلبات أولاً (عشان الـ Caching)
COPY requirements.txt .

# 5. تثبيت المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# 6. نسخ باقي ملفات المشروع (Dashboard, API, Models, Data)
COPY . .

# 7. فتح بورتات الـ Dashboard والـ API
EXPOSE 8501
EXPOSE 8000

# 8. الأمر النهائي (هنشغل الـ Dashboard والـ API مع بعض)
# ملاحظة: هنستخدم script بسيط أو نشغل الـ Dashboard حالياً كأولوية
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"] 
