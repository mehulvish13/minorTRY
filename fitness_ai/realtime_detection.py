from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent

# Load trained model and preprocessors from script directory.
model = joblib.load(ROOT_DIR / "exercise_model.pkl")
scaler = joblib.load(ROOT_DIR / "scaler.pkl")
label_encoder = joblib.load(ROOT_DIR / "label_encoder.pkl")
side_encoder = joblib.load(ROOT_DIR / "side_encoder.pkl")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return float(angle)


def normalize_exercise_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def select_body_side(landmarks):
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    return "left" if left_shoulder_x < right_shoulder_x else "right"


def lm_xy(landmarks, pose_landmark):
    lm = landmarks[pose_landmark.value]
    return [lm.x, lm.y]


def encode_side(side_name):
    classes = [str(c).lower() for c in side_encoder.classes_]
    desired = side_name.lower()

    if desired in classes:
        return int(side_encoder.transform([desired])[0]), desired, False

    fallback = classes[0]
    return int(side_encoder.transform([fallback])[0]), fallback, True


def build_features(landmarks, side_name):
    is_left = side_name == "left"

    shoulder_lm = mp_pose.PoseLandmark.LEFT_SHOULDER if is_left else mp_pose.PoseLandmark.RIGHT_SHOULDER
    elbow_lm = mp_pose.PoseLandmark.LEFT_ELBOW if is_left else mp_pose.PoseLandmark.RIGHT_ELBOW
    wrist_lm = mp_pose.PoseLandmark.LEFT_WRIST if is_left else mp_pose.PoseLandmark.RIGHT_WRIST
    hip_lm = mp_pose.PoseLandmark.LEFT_HIP if is_left else mp_pose.PoseLandmark.RIGHT_HIP
    knee_lm = mp_pose.PoseLandmark.LEFT_KNEE if is_left else mp_pose.PoseLandmark.RIGHT_KNEE
    ankle_lm = mp_pose.PoseLandmark.LEFT_ANKLE if is_left else mp_pose.PoseLandmark.RIGHT_ANKLE

    shoulder = lm_xy(landmarks, shoulder_lm)
    elbow = lm_xy(landmarks, elbow_lm)
    wrist = lm_xy(landmarks, wrist_lm)
    hip = lm_xy(landmarks, hip_lm)
    knee = lm_xy(landmarks, knee_lm)
    ankle = lm_xy(landmarks, ankle_lm)

    shoulder_angle = calculate_angle(elbow, shoulder, hip)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    ankle_angle = calculate_angle(knee, ankle, [ankle[0], ankle[1] + 0.1])

    shoulder_ground = shoulder_angle
    elbow_ground = elbow_angle
    hip_ground = hip_angle
    knee_ground = knee_angle
    ankle_ground = ankle_angle

    side_encoded, encoded_side_name, side_fallback_used = encode_side(side_name)

    features = [
        side_encoded,
        shoulder_angle,
        elbow_angle,
        hip_angle,
        knee_angle,
        ankle_angle,
        shoulder_ground,
        elbow_ground,
        hip_ground,
        knee_ground,
        ankle_ground,
    ]

    return features, elbow_angle, knee_angle, encoded_side_name, side_fallback_used


def main():
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and try again.")

    counter = 0
    stage = None
    pred_buffer = deque(maxlen=7)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            side_name = select_body_side(landmarks)
            features, elbow_angle, knee_angle, encoded_side_name, side_fallback_used = build_features(landmarks, side_name)
            pred = model.predict(scaler.transform([features]))
            exercise_raw = str(label_encoder.inverse_transform(pred)[0])

            pred_buffer.append(exercise_raw)
            exercise = Counter(pred_buffer).most_common(1)[0][0]
            normalized = normalize_exercise_name(exercise)

            if normalized in {"pushup", "pushups"}:
                if elbow_angle < 100:
                    stage = "down"
                if elbow_angle > 150 and stage == "down":
                    stage = "up"
                    counter += 1

            elif normalized in {"squat", "squats"}:
                if knee_angle < 90:
                    stage = "down"
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1

            cv2.putText(image, f"Exercise: {exercise}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(image, f"Side: {side_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
            if side_fallback_used:
                cv2.putText(
                    image,
                    f"Side fallback: using '{encoded_side_name}'",
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 120, 255),
                    2,
                )
            cv2.putText(image, f"Reps: {counter}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f"Elbow: {int(elbow_angle)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(image, f"Knee: {int(knee_angle)}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imshow("Fitness AI Trainer (press q to quit)", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()