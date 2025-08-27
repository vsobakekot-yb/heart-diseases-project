#  Heart Disease Prediction Service

Сервис для предсказания риска сердечно-сосудистых заболеваний на основе обученной модели машинного обучения.  
Реализован на **FastAPI**.

---

##  Структура проекта
├── app/
│   ├── api.py             # REST API (FastAPI)
│   ├── model_service.py   # Класс для работы с моделью
├── model.pkl              # Сериализованная модель (joblib)
├── test.csv               # Тестовые данные (используются ревьюером)
├── requirements.txt       # Зависимости проекта
└── README.md              # Документация

---

##  Установка и запуск

### 1. Клонирование и переход в папку проекта
```bash
git clone <repo_url>
cd <project_name>
```

---

## Установка зависимостей  

pip install -r requirements.txt

## Запуск сервиса

uvicorn app.api:app --reload

После запуска сервис будет доступен по адресу:
http://127.0.0.1:8000/docs

## API

 HealthCheck GET/ :

{
  "message": "Heart Disease Prediction Service is running"
}

POST /predict

{
  "predictions": [
    {
      "id": 7746,
      "prediction": 1,
      "probability": 0.42
    },
    {
      "id": 4202,
      "prediction": 0,
      "probability": 0.33
    }
  ]
}


