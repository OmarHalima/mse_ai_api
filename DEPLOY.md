# دليل النشر

## 1. PythonAnywhere (مجاني / مدفوع)

### الخطوات:
1. سجّل دخول على pythonanywhere.com
2. افتح **Bash console** وشغّل:
```bash
git clone https://github.com/YOUR_REPO/mse_ai_api.git
cd mse_ai_api
pip install -r requirements.txt
```

3. اذهب لـ **Web tab** → Add new web app → Manual configuration → Python 3.10

4. في **WSGI configuration file** احذف كل شيء وضع:
```python
import sys, os
sys.path.insert(0, '/home/YOUR_USERNAME/mse_ai_api')
from main import app as application
```

5. في **Environment variables** أضف:
```
API_SECRET_KEY = your-secret-key
CHATGPT_ACCESS_TOKEN = your-token
```

6. اضغط **Reload** ✓

الـ API سيكون على: `https://YOUR_USERNAME.pythonanywhere.com`

---

## 2. VPS / Docker

```bash
git clone https://github.com/YOUR_REPO/mse_ai_api.git
cd mse_ai_api

# عدّل المتغيرات في docker-compose.yml
docker-compose up -d --build
```

---

## 3. الحصول على ChatGPT Access Token

1. افتح المتصفح وسجّل دخول على chatgpt.com
2. اذهب لـ: `https://chatgpt.com/api/auth/session`
3. انسخ قيمة `accessToken`
4. ضعها في لوحة التحكم: `YOUR_DOMAIN/dashboard` → تبويب الاعتمادات

---

## 4. إعداد n8n

- Base URL: `https://YOUR_DOMAIN/v1`
- API Key: قيمة `API_SECRET_KEY`
