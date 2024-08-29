
import os
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# Construct the path to the pickle files
model_filename = os.path.join(current_dir, 'model_try.pkl')
scaler_filename = os.path.join(current_dir, 'scaler_try.pkl')


with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

def extract_landmarks(results):
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

def predict_with_confidence(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = np.max(probabilities)
    return prediction, confidence

