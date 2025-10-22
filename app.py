import os

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
pipe = joblib.load("model/xgb_model.pkl")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class PredictionRequest(BaseModel):
    Time_spent_Alone: int
    Stage_fear: str
    Social_event_attendance: int
    Going_outside: int
    Drained_after_socializing: str
    Friends_circle_size: int
    Post_frequency: int


@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    print("Validated data:", req.model_dump())
    input_data = pd.DataFrame([req.model_dump()])
    pred_num = pipe.predict(input_data)
    label_map = {0: "Extrovert", 1: "Introvert"}  # must match your training mapping
    pred_labels = [label_map.get(int(p), "Unknown") for p in pred_num]

    # If you want confidence (probability) shown as well:
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(input_data)
        confidences = probs.max(axis=1)
        print(
            [f"{lab} (conf={conf:.2f})" for lab, conf in zip(pred_labels, confidences)]
        )
    else:
        confidences = [None]

    return {"prediction": pred_labels[0], "confidence": float(confidences[0])}
