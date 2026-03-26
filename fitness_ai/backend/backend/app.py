from pathlib import Path
from typing import Union

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[2]
model = joblib.load(ROOT_DIR / "exercise_model.pkl")
scaler = joblib.load(ROOT_DIR / "scaler.pkl")
label_encoder = joblib.load(ROOT_DIR / "label_encoder.pkl")


class PredictRequest(BaseModel):
    data: list[float]

@app.get("/")
def healthcheck():
    return {"status": "ok"}


@app.get("/ui")
def ui():
    ui_path = Path(__file__).resolve().parent / "frontend" / "index.html"
    return FileResponse(ui_path)


@app.post("/predict")
def predict(payload: Union[list[float], PredictRequest] = Body(...)):
    features = payload if isinstance(payload, list) else payload.data

    if not features:
        raise HTTPException(status_code=400, detail="Input data cannot be empty")

    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Input data must be numeric")

    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is not None and arr.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} values, got {arr.shape[1]}"
        )

    try:
        pred = model.predict(scaler.transform(arr))
        exercise = label_encoder.inverse_transform(pred)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return {"exercise": str(exercise)}