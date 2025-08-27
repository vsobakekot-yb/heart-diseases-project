import joblib
import pandas as pd
import numpy as np


class HeartAttackModel:
    """
    Класс для загрузки и применения модели.
    """

    def __init__(self, model_path: str, threshold: float = 0.40):
        # Загружаем пайплайн (preprocessor + RandomForest)
        self.model = joblib.load(model_path)
        self.threshold = threshold

        # Попробуем достать список признаков, которые модель ждёт
        try:
            self.feature_names = list(
                self.model.named_steps["preprocessor"].feature_names_in_
            )
        except Exception:
            # fallback если пайплайн не поддерживает .feature_names_in_
            self.feature_names = None
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка входных данных: нормализация имён колонок,
        удаление/добавление недостающих, кодировка категориальных признаков.
        """
        print("\n=== Колонки на входе ===")
        print(df.columns.tolist())

        # Приводим имена к snake_case
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        print("\n=== После нормализации имён ===")
        print(df.columns.tolist())

        # ---- Кодировка категориальных признаков ----
        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(df["gender"])

        if "smoking" in df.columns:
            df["smoking"] = df["smoking"].map({"Yes": 1, "No": 0}).fillna(df["smoking"])

        if "diet" in df.columns:
            df["diet"] = df["diet"].map({"Healthy": 1, "Unhealthy": 0}).fillna(df["diet"])

        # ---- Работа с колонками ----
        if self.feature_names is not None:
            # оставляем только те, что в модели
            df = df.loc[:, [c for c in df.columns if c in self.feature_names]]

            # добавляем отсутствующие
            for col in self.feature_names:
                if col not in df.columns:
                    print(f"[WARN] Добавляю отсутствующую колонку: {col}")
                    df[col] = 0

            # упорядочиваем
            df = df[self.feature_names]

        print("\n=== Колонки после _prepare ===")
        print(df.columns.tolist())

        return df

    def predict(self, df: pd.DataFrame):
        """
        Делает предсказания для DataFrame с признаками пациентов.
        Возвращает список (preds, proba).
        """
        df = self._prepare(df)

        proba = self.model.predict_proba(df)[:, 1]
        preds = (proba >= self.threshold).astype(int)

        return preds.tolist(), proba.tolist()