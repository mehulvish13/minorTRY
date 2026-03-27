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


def normalize_exercise_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def assess_form(exercise_name: str, features: np.ndarray) -> dict:
    # Feature order used by the model:
    # [side, shoulder, elbow, hip, knee, ankle, shoulder_g, elbow_g, hip_g, knee_g, ankle_g]
    elbow_angle = float(features[2])
    hip_angle = float(features[3])
    knee_angle = float(features[4])

    normalized = normalize_exercise_name(exercise_name)
    feedback: list[str] = []

    if normalized in {"pushup", "pushups"}:
        if elbow_angle > 110:
            feedback.append("Go lower in the push-up")
        if hip_angle < 150 or hip_angle > 195:
            feedback.append("Keep your body straight")

    elif normalized in {"squat", "squats"}:
        if knee_angle > 120:
            feedback.append("Go deeper in the squat")
        if hip_angle < 55 or hip_angle > 170:
            feedback.append("Keep your chest up")

    status = "correct" if not feedback else "incorrect"
    score = max(0, 100 - (len(feedback) * 25))

    return {
        "form_status": status,
        "form_score": score,
        "feedback": feedback,
        "angles": {
            "elbow": round(elbow_angle, 1),
            "hip": round(hip_angle, 1),
            "knee": round(knee_angle, 1),
        },
    }

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

    form_result = assess_form(str(exercise), arr[0])

    return {
        "exercise": str(exercise),
        **form_result,
    }