import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
DATA_PATH = Path(__file__).resolve().parent / "exercise_angles.csv"
data = pd.read_csv(DATA_PATH)

# Remove spaces in column names (safety)
data.columns = data.columns.str.strip()

# -----------------------------
# Encode 'Side' column
# -----------------------------
side_encoder = LabelEncoder()
data["Side"] = side_encoder.fit_transform(data["Side"])

# -----------------------------
# Encode Label column
# -----------------------------
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# -----------------------------
# Features and target
# -----------------------------
X = data.drop("Label", axis=1)
y = data["Label"]

# -----------------------------
# Normalize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# -----------------------------
# Save model and preprocessors
# -----------------------------
joblib.dump(model, "exercise_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(side_encoder, "side_encoder.pkl")
print(data["Side"].unique())

print("Model and encoders saved successfully.")