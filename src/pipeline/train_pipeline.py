import os
from pathlib import Path

import joblib
import pandas as pd
from src.pipeline.preprocessing_pipeline import build_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

if __name__ == "__main__":
    preprocessor = build_preprocessing()
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "processed" / "cleaned.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        [
            ("process", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=25, max_depth=3, learning_rate=0.1, random_state=42
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    model_path = os.path.join(project_root, "model", "xgb_model.pkl")

    joblib.dump(pipe, model_path)

    print("Model training completed")
    # # ensure model output directory exists and save with a resolved path
    # model_dir = project_root / "model"
    # model_dir.mkdir(parents=True, exist_ok=True)
    # model_path = model_dir / "xgb_model.pkl"
    # joblib.dump(pipe, model_path)
    # print(f"Model training completed. Saved to {model_path}")
