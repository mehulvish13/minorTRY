#  Fitness Exercise Predictor

Detect exercise type from pose features, run real-time webcam inference with rep counting, and expose predictions through a FastAPI backend.

## Features
- Train a classification model from CSV pose-angle features
- Run offline predictions on dataset rows
- Detect exercises in real time from webcam feed
- Count reps and provide basic form feedback
- Test model inference via API and browser UI

## Project Structure
```text
fitness_ai/
  train_model.py
  predict.py
  realtime_detection.py
  exercise_angles.csv
  backend/
    backend/
      app.py
      frontend/
        index.html
run.bat
requirements.txt
```

## Requirements
- Python 3.11 (recommended)
- Webcam (for realtime mode)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup
Open a terminal in the repository root (where `run.bat` exists).

### Windows (PowerShell)
```powershell
py -3.11 -m venv fitness_ai/.venv311
.\fitness_ai\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3.11 -m venv fitness_ai/.venv311
source fitness_ai/.venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is unavailable, use your installed Python version.

## Quick Start

### 1) Train the model
```bash
cd fitness_ai
python train_model.py
```

Generated artifacts:
- `exercise_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `side_encoder.pkl`

### 2) Run offline prediction
```bash
cd fitness_ai
python predict.py
```

### 3) Run realtime webcam detection
Windows shortcut:
```bat
run.bat
```

Cross-platform:
```bash
cd fitness_ai
python realtime_detection.py
```

Control: press `q` to quit.

### 4) Run backend API and UI
```bash
cd fitness_ai
python -m uvicorn backend.backend.app:app --reload
```

Open:
- Health: `http://127.0.0.1:8000/`
- UI: `http://127.0.0.1:8000/ui`
- Predict endpoint: `POST http://127.0.0.1:8000/predict`

## API Example

### Request
```json
{
  "data": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}
```

### Response (example)
```json
{
  "exercise": "pushup",
  "form_status": "correct",
  "form_score": 100,
  "feedback": [],
  "angles": {
    "elbow": 85.2,
    "hip": 170.3,
    "knee": 165.8
  }
}
```

Note: Input `data` length must match the model feature size.

## Troubleshooting
- `ModuleNotFoundError`: activate your virtual environment, then reinstall dependencies.
- Webcam not opening: check camera permissions and close other camera apps.
- Feature length error in `/predict`: provide the expected number of numeric values.
- Missing model files: run `python train_model.py` first.

## Notes
- `fitness_ai/check_dataset.py` uses an old local path and is not required for normal usage.
- Keep generated `.pkl` files in `fitness_ai/` so scripts can load them.