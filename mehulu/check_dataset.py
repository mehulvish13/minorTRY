import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\fitnessapp\exercise_angles.csv")

print(df.head())

# Remove Side column
df = df.drop("Side", axis=1)

# Encode labels
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Split features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "exercise_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

print("Model and preprocessing objects saved successfully")