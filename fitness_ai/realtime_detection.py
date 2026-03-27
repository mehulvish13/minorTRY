from collections import Counter, deque
from pathlib import Path
import time

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

MIN_VISIBILITY = 0.60
SMOOTH_ALPHA = 0.35
STAGE_MIN_FRAMES = 3
MIN_REP_DURATION_SEC = 0.60
MAX_REP_DURATION_SEC = 6.00


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


def smooth_angle(previous, current, alpha=SMOOTH_ALPHA):
    if previous is None:
        return float(current)
    return float((alpha * current) + ((1.0 - alpha) * previous))


def has_required_visibility(landmarks, side_name, threshold=MIN_VISIBILITY):
    is_left = side_name == "left"
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER if is_left else mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW if is_left else mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST if is_left else mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP if is_left else mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE if is_left else mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE if is_left else mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]
    return all(landmarks[lm.value].visibility >= threshold for lm in required)


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

    return features, elbow_angle, knee_angle, hip_angle, encoded_side_name, side_fallback_used


def init_rep_metrics(start_time=None):
    return {
        "start_time": time.time() if start_time is None else start_time,
        "min_elbow": float("inf"),
        "max_elbow": float("-inf"),
        "min_knee": float("inf"),
        "max_knee": float("-inf"),
        "min_hip": float("inf"),
        "max_hip": float("-inf"),
    }


def update_rep_metrics(metrics, elbow_angle, knee_angle, hip_angle):
    metrics["min_elbow"] = min(metrics["min_elbow"], elbow_angle)
    metrics["max_elbow"] = max(metrics["max_elbow"], elbow_angle)
    metrics["min_knee"] = min(metrics["min_knee"], knee_angle)
    metrics["max_knee"] = max(metrics["max_knee"], knee_angle)
    metrics["min_hip"] = min(metrics["min_hip"], hip_angle)
    metrics["max_hip"] = max(metrics["max_hip"], hip_angle)


def evaluate_pushup_form(metrics, duration_sec):
    feedback = []
    passed = {
        "depth": metrics["min_elbow"] <= 95,
        "lockout": metrics["max_elbow"] >= 155,
        "body_line": metrics["min_hip"] >= 150 and metrics["max_hip"] <= 195,
        "tempo": 0.8 <= duration_sec <= 4.0,
    }

    if not passed["depth"]:
        feedback.append("Go lower")
    if not passed["lockout"]:
        feedback.append("Fully extend at the top")
    if not passed["body_line"]:
        feedback.append("Keep your body straight")
    if not passed["tempo"]:
        feedback.append("Control your speed")

    score = 100
    score -= 25 if not passed["depth"] else 0
    score -= 25 if not passed["lockout"] else 0
    score -= 25 if not passed["body_line"] else 0
    score -= 10 if not passed["tempo"] else 0
    score = max(0, score)

    status = "CORRECT" if not feedback else "INCORRECT"
    return status, score, feedback


def evaluate_squat_form(metrics, duration_sec):
    feedback = []
    passed = {
        "depth": metrics["min_knee"] <= 95,
        "lockout": metrics["max_knee"] >= 165,
        "torso": metrics["min_hip"] >= 55 and metrics["max_hip"] <= 170,
        "tempo": 0.8 <= duration_sec <= 4.0,
    }

    if not passed["depth"]:
        feedback.append("Go deeper")
    if not passed["lockout"]:
        feedback.append("Stand fully at the top")
    if not passed["torso"]:
        feedback.append("Keep chest up and reduce forward lean")
    if not passed["tempo"]:
        feedback.append("Control your speed")

    score = 100
    score -= 25 if not passed["depth"] else 0
    score -= 25 if not passed["lockout"] else 0
    score -= 25 if not passed["torso"] else 0
    score -= 10 if not passed["tempo"] else 0
    score = max(0, score)

    status = "CORRECT" if not feedback else "INCORRECT"
    return status, score, feedback


def is_valid_pushup_rep(metrics, duration_sec):
    return (
        MIN_REP_DURATION_SEC <= duration_sec <= MAX_REP_DURATION_SEC
        and metrics["min_elbow"] <= 100
        and metrics["max_elbow"] >= 150
    )


def is_valid_squat_rep(metrics, duration_sec):
    return (
        MIN_REP_DURATION_SEC <= duration_sec <= MAX_REP_DURATION_SEC
        and metrics["min_knee"] <= 95
        and metrics["max_knee"] >= 160
    )


def main():
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and try again.")

    counter = 0
    stage = None
    pred_buffer = deque(maxlen=7)
    rep_metrics = init_rep_metrics()
    form_status = "N/A"
    form_score = 0
    form_feedback = []
    smoothed_angles = {"elbow": None, "knee": None, "hip": None}
    down_frames = 0
    up_frames = 0
    visibility_warn = ""
    exercise = "N/A"
    last_normalized = None

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
            visibility_ok = has_required_visibility(landmarks, side_name)
            features, elbow_angle, knee_angle, hip_angle, encoded_side_name, side_fallback_used = build_features(landmarks, side_name)

            elbow_angle = smooth_angle(smoothed_angles["elbow"], elbow_angle)
            knee_angle = smooth_angle(smoothed_angles["knee"], knee_angle)
            hip_angle = smooth_angle(smoothed_angles["hip"], hip_angle)
            smoothed_angles["elbow"] = elbow_angle
            smoothed_angles["knee"] = knee_angle
            smoothed_angles["hip"] = hip_angle

            if not visibility_ok:
                visibility_warn = "Low landmark confidence - hold camera steady"
                down_frames = 0
                up_frames = 0
            else:
                visibility_warn = ""

            features[2] = elbow_angle
            features[3] = hip_angle
            features[4] = knee_angle
            features[7] = elbow_angle
            features[8] = hip_angle
            features[9] = knee_angle

            pred = model.predict(scaler.transform([features]))
            exercise_raw = str(label_encoder.inverse_transform(pred)[0])

            pred_buffer.append(exercise_raw)
            exercise = Counter(pred_buffer).most_common(1)[0][0]
            normalized = normalize_exercise_name(exercise)

            if last_normalized is not None and normalized != last_normalized:
                stage = None
                down_frames = 0
                up_frames = 0
                rep_metrics = init_rep_metrics()
            last_normalized = normalized

            if visibility_ok:
                update_rep_metrics(rep_metrics, elbow_angle, knee_angle, hip_angle)

            if normalized in {"pushup", "pushups"}:
                if visibility_ok:
                    if elbow_angle < 100:
                        down_frames += 1
                    else:
                        down_frames = 0

                    if elbow_angle > 150:
                        up_frames += 1
                    else:
                        up_frames = 0

                    if stage != "down" and down_frames >= STAGE_MIN_FRAMES:
                        stage = "down"
                        rep_metrics = init_rep_metrics()

                    if stage == "down" and up_frames >= STAGE_MIN_FRAMES:
                        duration = time.time() - rep_metrics["start_time"]
                        if is_valid_pushup_rep(rep_metrics, duration):
                            stage = "up"
                            counter += 1
                            form_status, form_score, form_feedback = evaluate_pushup_form(rep_metrics, duration)
                        else:
                            form_status = "INCORRECT"
                            form_score = 0
                            form_feedback = ["Complete full range with stable tempo"]
                        rep_metrics = init_rep_metrics()
                        down_frames = 0
                        up_frames = 0

            elif normalized in {"squat", "squats"}:
                if visibility_ok:
                    if knee_angle < 90:
                        down_frames += 1
                    else:
                        down_frames = 0

                    if knee_angle > 160:
                        up_frames += 1
                    else:
                        up_frames = 0

                    if stage != "down" and down_frames >= STAGE_MIN_FRAMES:
                        stage = "down"
                        rep_metrics = init_rep_metrics()

                    if stage == "down" and up_frames >= STAGE_MIN_FRAMES:
                        duration = time.time() - rep_metrics["start_time"]
                        if is_valid_squat_rep(rep_metrics, duration):
                            stage = "up"
                            counter += 1
                            form_status, form_score, form_feedback = evaluate_squat_form(rep_metrics, duration)
                        else:
                            form_status = "INCORRECT"
                            form_score = 0
                            form_feedback = ["Complete full range with stable tempo"]
                        rep_metrics = init_rep_metrics()
                        down_frames = 0
                        up_frames = 0

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
            cv2.putText(image, f"Hip: {int(hip_angle)}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            status_color = (0, 200, 0) if form_status == "CORRECT" else (0, 0, 255)
            cv2.putText(image, f"Form: {form_status} ({form_score})", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            if form_feedback:
                cv2.putText(image, f"Tip: {form_feedback[0]}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
            if visibility_warn:
                cv2.putText(image, visibility_warn, (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        cv2.imshow("Fitness AI Trainer (press q to quit)", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()