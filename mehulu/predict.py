import joblib
import pandas as pd

# Load model and preprocessing objects
model = joblib.load("exercise_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
side_encoder = joblib.load("side_encoder.pkl")

# Load dataset
df = pd.read_csv("exercise_angles.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Encode 'Side' column
df["Side"] = side_encoder.transform(df["Side"])

# Take one sample (remove Label column)
sample = df.drop("Label", axis=1).iloc[0]

# Convert to DataFrame
sample_df = pd.DataFrame([sample])

# Scale features
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_scaled)

# Convert number → exercise name
exercise_name = label_encoder.inverse_transform(prediction)

print("Predicted Exercise:", exercise_name[0])