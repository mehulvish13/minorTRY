# Mehulu Fitness Exercise Predictor - Project Report

## Chapter 1. Introduction

### 1.1 Introduction to Project
This project, **Mehulu Fitness Exercise Predictor**, is an AI-enabled fitness monitoring system that detects exercises and evaluates form using human pose landmarks. It supports model training from datasets, offline prediction, and real-time webcam-based exercise detection with repetition counting. A FastAPI backend and browser UI are also provided.

### 1.2 Project Category
- **Application or System Development** (Internet-enabled through API)
- Also suitable as a **Research-based** mini project (ML + pose estimation)

### 1.3 Objectives
1. Build an ML model to classify exercises from pose-angle features.
2. Provide real-time exercise prediction using webcam input.
3. Count repetitions automatically.
4. Assess form and provide feedback.
5. Expose prediction via REST API and web UI.
6. Keep the system lightweight and low-cost.

### 1.4 Problem Formulation
People often exercise without supervision, causing incorrect form and potential injury. Professional coaching is not always available. A low-cost automated system is needed to detect exercise type, count repetitions, and provide immediate corrective feedback using only a camera.

### 1.5 Identification/Reorganization of Need
**Identified Needs**
1. Real-time exercise recognition.
2. No special hardware dependency.
3. Basic form correction.
4. Easy setup for students and home users.
5. API access for future integration.

**Reorganized as System Needs**
1. Pose extraction module.
2. Feature engineering module.
3. ML prediction module.
4. Repetition counting and feedback module.
5. API + UI module.

### 1.6 Existing System
1. Manual counting and mirror-based correction.
2. Generic workout videos without personalized feedback.
3. Wearables that track motion but not detailed posture quality.

**Limitations**
- No low-cost, real-time form guidance.
- Limited personalization.
- Less suitable for academic ML extension.

### 1.7 Proposed System
The proposed system combines MediaPipe pose extraction with a trained Random Forest classifier. It computes joint-angle features, predicts exercise class, tracks stage transitions for rep counting, and applies rule-based form evaluation. FastAPI provides web/API access.

### 1.8 Unique Features of the System
1. Real-time webcam exercise prediction.
2. Repetition counting with stage logic.
3. Form scoring and textual feedback.
4. API endpoint for integration.
5. Browser test UI.
6. No expensive hardware required.

---

## Chapter 2. Requirement Analysis and System Specification

### 2.1 Feasibility Study
#### Technical Feasibility
- Uses stable libraries: MediaPipe, OpenCV, scikit-learn, FastAPI.
- Runs on standard laptop hardware with webcam.
- Uses efficient tabular features for prediction.

#### Economical Feasibility
- Fully open-source stack.
- No paid cloud required.
- Existing hardware is sufficient.

#### Operational Feasibility
- Simple setup and execution flow.
- Works through script and API/UI modes.

### 2.2 Software Requirement Specification (SRS)

#### Data Requirement
1. CSV data with labeled exercise-angle features.
2. Real-time webcam landmark stream.
3. Saved model artifacts (`exercise_model.pkl`, `scaler.pkl`, encoders).

#### Functional Requirement
1. Train model from dataset.
2. Predict exercise from feature input.
3. Perform real-time webcam detection.
4. Count reps and evaluate form.
5. Expose `/predict`, `/ui`, and health endpoints.

#### Performance Requirement
1. Real-time inference should remain interactive.
2. API response should be quick for single-request prediction.
3. Model loading should occur at startup.

#### Dependability Requirement
1. Validate non-empty numeric input.
2. Check input feature length.
3. Handle camera access errors clearly.

#### Maintainability Requirement
1. Modular Python scripts.
2. Named constants for thresholds.
3. Easy extension for new exercise classes.

#### Security Requirement
1. Input validation at API boundary.
2. Safe exception handling.
3. Localhost deployment by default.

#### Look and Feel Requirement
1. Simple and readable browser UI.
2. Clear response formatting.
3. Useful overlays in realtime video.

### 2.3 Validation
1. Train-test split accuracy evaluation.
2. Offline sample prediction validation.
3. API positive and negative test cases.
4. Realtime behavior testing with manual trials.

### 2.4 Expected Hurdles
1. Landmark noise due to lighting and camera angle.
2. False rep counts on partial motion.
3. Dataset imbalance.
4. Threshold sensitivity for form scoring.

### 2.5 SDLC Model to be Used
**Iterative Incremental Model**
- Suitable for progressive feature additions and repeated model/threshold refinement.

---

## Chapter 3. System Design

### 3.1 Design Approach
Hybrid modular approach with object-oriented and function-oriented decomposition.

### 3.2 Detail Design
Main modules:
1. Data Preparation and Training
2. Offline Prediction
3. Realtime Detection and Rep Counter
4. Form Assessment
5. Backend API
6. Frontend UI

### 3.3 Structured Analysis and Design Tools

#### DFD (Level 0)
- Input: webcam frames or numeric features
- Process: fitness prediction engine
- Output: exercise label, rep count, form feedback

#### DFD (Level 1)
1. Capture landmarks
2. Build angle features
3. Scale/encode input
4. Predict exercise
5. Evaluate form and count reps
6. Return/display response

#### Data Dictionary (Core Fields)
- `side`
- `shoulder_angle`, `elbow_angle`, `hip_angle`, `knee_angle`, `ankle_angle`
- `label`
- `form_status`, `form_score`, `feedback`

#### Structured Chart
1. Main Controller
2. Pose Processing
3. Feature Engineering
4. Prediction Engine
5. Feedback/Counter Logic
6. API/UI Layer

#### Flowchart (Text)
Start -> Capture -> Extract -> Feature Build -> Scale -> Predict -> Assess -> Display/Return -> Loop/End

#### UML (Conceptual)
- Predictor
- FormAssessor
- RealtimeDetector
- APIController

### 3.4 User Interface Design
1. Browser UI accepts feature values and shows API result.
2. Realtime window overlays skeleton, prediction, counter, and feedback.

### 3.5 Database Design
Current implementation uses CSV + serialized model files (no persistent RDBMS).

### 3.6 ER Diagram (Proposed Extension)
Entities:
1. User
2. WorkoutSession
3. ExerciseRecord
4. FormFeedback

### 3.7 Normalization (Proposed DB)
1. 1NF: atomic attributes
2. 2NF: separate session and record data
3. 3NF: move feedback details to dedicated table

### 3.8 Database Manipulation
1. Insert session record
2. Insert prediction/form logs
3. Query progress history
4. Update session summary

### 3.9 Database Connection Controls and Strings
Use environment-based secure connection strings for SQLite/PostgreSQL in future extension.

### 3.10 Methodology
1. Dataset inspection
2. Feature preprocessing
3. Model training and evaluation
4. Artifact saving
5. Realtime and API integration
6. Iterative testing and tuning

---

## Chapter 4. Implementation, Testing, and Maintenance

### 4.1 Languages, IDEs, Tools, Technologies
- Python
- VS Code
- NumPy, Pandas, scikit-learn, Joblib
- MediaPipe, OpenCV
- FastAPI, Uvicorn, Pydantic

### 4.2 Coding Standards
1. PEP 8 style conventions
2. Modular code organization
3. Clear naming and validation logic
4. Reusable constants for thresholds

### 4.3 Project Scheduling (PERT/GANTT Style Plan)
1. Week 1: Requirement analysis
2. Week 2: Data preparation
3. Week 3: Model training
4. Week 4: Realtime integration
5. Week 5: API/UI integration
6. Week 6: Testing and documentation

### 4.4 Testing Techniques and Test Plans
#### Unit Testing
- Angle calculations
- Feature generation
- Input validation

#### Integration Testing
- Scaler/model/encoder compatibility
- API endpoint behavior
- UI request-response flow

#### System Testing
- End-to-end realtime prediction
- Rep counting reliability
- Feedback quality under posture variations

### 4.5 Maintenance
1. Retrain model with new labeled data.
2. Tune thresholds and rules over time.
3. Add new exercises incrementally.
4. Keep dependencies updated and tested.

---

## Chapter 5. Results and Discussions

### 5.1 User Interface Representation
The project provides:
1. Realtime webcam interface with overlays.
2. Browser UI for API testing.

### 5.2 Brief Description of Modules
1. Training module
2. Offline prediction module
3. Realtime detection module
4. Form analysis module
5. Backend and UI module

### 5.3 Snapshots of System with Brief Details
Include report snapshots for:
1. Training output and accuracy.
2. Realtime pose + prediction screen.
3. Rep counting and form feedback display.
4. API success response in browser UI.
5. API validation error response.

### 5.4 Back End Representation (Database)
Current backend is stateless and model-file based. For production scaling, use SQLite/PostgreSQL with workout logs.

### 5.5 Snapshots of Database Tables with Brief Description
If DB extension is implemented, include snapshots of:
1. Users
2. WorkoutSessions
3. ExerciseRecords
4. FeedbackLogs

### Discussion
The system demonstrates that a lightweight ML + pose-estimation pipeline can provide practical real-time exercise classification and form feedback. It is modular, low-cost, and suitable for further extension into multi-user tracking and analytics.