import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Define project root relative to this script
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "cleaned.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Split data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model
    model_path = project_root / "model" / "xgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipe = joblib.load(model_path)
    print("Model loaded successfully")

    # Predictions
    y_pred = pipe.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Accuracy Score:", accuracy)

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix for Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_file_path = "confusion_matrix_model.png"
    plt.savefig(cm_file_path)
    plt.close()

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_artifact(cm_file_path)

    # Save metrics to JSON
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    with open("metrics.json", "w") as file:
        json.dump(metrics_dict, file, indent=4)
