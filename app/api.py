from fastapi import FastAPI, UploadFile, File
import pandas as pd
from app.model_service import HeartAttackModel

app = FastAPI(title="Heart Attack Risk Prediction API")

# загружаем модель при старте
model = HeartAttackModel("model.pkl", threshold=0.40)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает CSV с тестовой выборкой, возвращает JSON с предсказаниями.
    """
    df = pd.read_csv(file.file)

    preds, proba = model.predict(df)

    ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))

    return {
        "predictions": [
            {"id": i, "prediction": p, "probability": round(pr, 3)}
            for i, p, pr in zip(ids, preds, proba)
        ]
    }