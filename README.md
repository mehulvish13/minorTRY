# Mehulu Fitness Exercise Predictor

This project detects workout type from pose features and supports:

- model training from CSV data,
- offline prediction on dataset rows,
- realtime webcam exercise detection with rep counting,
- a FastAPI backend with a browser test UI.

## Project layout

```text
fitness_ai/
  train_model.py
  predict.py
  realtime_detection.py
  exercise_angles.csv
  backend/backend/app.py
  backend/backend/frontend/index.html
run.bat
```

## 1) Requirements

- Python 3.11 recommended
- Webcam (for realtime mode)

Python packages used by this project:

- numpy
- pandas
- scikit-learn
- joblib
- mediapipe
- opencv-python
- fastapi
- uvicorn
- pydantic

## 2) Setup (works anywhere)

Open a terminal in the repository root (the folder that contains `run.bat`).

### Windows (PowerShell)

```powershell
py -3.11 -m venv fitness_ai/.venv311
.\fitness_ai\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn joblib mediapipe opencv-python fastapi uvicorn pydantic
```

### macOS / Linux (bash/zsh)

```bash
python3.11 -m venv fitness_ai/.venv311
source fitness_ai/.venv311/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn joblib mediapipe opencv-python fastapi uvicorn pydantic
```

If your system does not have `python3.11`, use your available `python3` version and keep the same virtual environment folder name.

## 3) Train the model

From the repository root:

```bash
cd fitness_ai
python train_model.py
```

This generates:

- `exercise_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `side_encoder.pkl`

## 4) Run offline prediction test

```bash
cd fitness_ai
python predict.py
```

## 5) Run realtime webcam detection

### Easiest on Windows

```bat
run.bat
```

### Cross-platform command

```bash
cd fitness_ai
python realtime_detection.py
```

Controls:

- Press `q` to quit realtime window.

## 6) Run backend API + browser UI

From repository root:

```bash
cd fitness_ai
python -m uvicorn backend.backend.app:app --reload
```

Open in browser:

- API health check: `http://127.0.0.1:8000/`
- Test UI: `http://127.0.0.1:8000/ui`
- Predict endpoint: `POST http://127.0.0.1:8000/predict`

### `/predict` request format

JSON body:

```json
{
  "data": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}
```

The number of values must match the model feature count.

## 7) Troubleshooting

- `ModuleNotFoundError`: activate the virtual environment and reinstall packages.
- Webcam not opening: check camera permissions and close other apps using camera.
- Feature size error in `/predict`: ensure your `data` length matches model input size.
- Model files missing: run `python train_model.py` first.

## Notes

- The script `check_dataset.py` contains an old hardcoded local file path and is not required for normal use.
- Keep model `.pkl` files in `fitness_ai/` so all scripts can load them correctly.