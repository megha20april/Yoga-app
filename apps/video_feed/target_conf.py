import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time
import pyttsx3

# Load the saved model and scaler
with open(r'C:\Users\Dell\OneDrive\Desktop\SIH\models\threeModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open(r'C:\Users\Dell\OneDrive\Desktop\SIH\models\threeScaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initializing variables for feedback mechanism
last_feedback_time = 0
last_feedback_confidence = 0
high_confidence_spoken = False

# Cooldown period for feedback
COOLDOWN_PERIOD = 5  # seconds

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6  # Increased for more reliable predictions

# Weighted confidence variables
confidence_window = deque(maxlen=10)  # Store the last 10 confidence values

# List of yoga poses and their instructions
yoga_poses = [
    "Chair Pose (Utkatasana)",
    "Cobra Pose (Bhujangasana)",
    "Tree Pose (Vrikshasana)",
]

def extract_landmarks(results):
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Pose tracking
pose_tracker = deque(maxlen=15)
current_pose = None
pose_start_time = None
pose_hold_time = 0
rep_count = 0

# Smoothing filter
alpha = 0.3  # Adjusted for more responsive smoothing
smoothed_landmarks = None

# Pose hold time threshold
POSE_HOLD_THRESHOLD = 5  # Increased to 5 seconds for a more challenging hold

# Last detected pose
last_detected_pose = None

# This variable will be set by the web interface
target_pose = "Chair Pose (Utkatasana)" 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect poses
    results = pose.process(image)
    
    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Extract landmarks
    landmarks = extract_landmarks(results)
    
    if landmarks is not None:
        # Apply smoothing filter
        if smoothed_landmarks is None:
            smoothed_landmarks = landmarks
        else:
            smoothed_landmarks = alpha * landmarks + (1 - alpha) * smoothed_landmarks
        
        # Scale the landmarks
        landmarks_scaled = scaler.transform([smoothed_landmarks])
        
        # Make prediction
        prediction = model.predict(landmarks_scaled)[0]
        confidence = np.max(model.predict_proba(landmarks_scaled))
        
        # Update confidence window
        confidence_window.append(confidence)
        
        # Calculate weighted confidence
        weighted_confidence = np.average(confidence_window, weights=range(1, len(confidence_window) + 1))
        
        # Check if the detected pose matches the target pose
        if weighted_confidence > CONFIDENCE_THRESHOLD:
            pose_tracker.append(prediction)
            if len(pose_tracker) == pose_tracker.maxlen:
                current_pose = max(set(pose_tracker), key=pose_tracker.count)
                
                # Map the detected pose to the yoga pose list
                matched_pose = False
                for yoga_pose in yoga_poses:
                    if current_pose.lower() in yoga_pose.lower():
                        current_pose = yoga_pose
                        matched_pose = True
                        break
                if not matched_pose:
                    current_pose = "Unknown Pose"
                
                # Proceed only if the detected pose matches the target pose
                if target_pose.lower() in current_pose.lower():
                    print(f"Target pose detected: {current_pose}")
                    last_detected_pose = current_pose

                    # Display the prediction and confidence
                    cv2.putText(image, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Confidence: {weighted_confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Timer for pose hold
                    if current_pose in yoga_poses and weighted_confidence > CONFIDENCE_THRESHOLD:
                        if pose_start_time is None:
                            pose_start_time = time.time()
                        pose_hold_time = time.time() - pose_start_time
                        cv2.putText(image, f"Hold Time: {pose_hold_time:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                    # Rep counter
                    if current_pose in yoga_poses and pose_hold_time > POSE_HOLD_THRESHOLD:
                        rep_count += 1
                        pose_start_time = None  # Reset timer
                    cv2.putText(image, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print(f"Detected pose '{current_pose}' does not match target '{target_pose}'. Ignoring...")
                    cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Feedback mechanism
    current_time = time.time()

    if weighted_confidence <= 0.5:
        if current_time - last_feedback_time >= COOLDOWN_PERIOD:  # Check if cooldown period has passed
            print("Pose not right. Try again Please.")
            '''engine.say("Pose not right. Try again, please.")
            engine.runAndWait()'''
            last_feedback_time = current_time
            last_feedback_confidence = weighted_confidence
            high_confidence_spoken = False
        elif weighted_confidence > last_feedback_confidence:
            print("That's better! Keep improving.")
            '''engine.say("That's better! Keep improving.")
            engine.runAndWait()'''
            last_feedback_time = current_time
            last_feedback_confidence = weighted_confidence
    elif 0.5 < weighted_confidence < 0.8:
        if current_time - last_feedback_time >= COOLDOWN_PERIOD:  # Check if cooldown period has passed
            print("Good going. Keep trying!")
            '''engine.say("Good going. Keep trying!")
            engine.runAndWait()'''
            last_feedback_time = current_time
            last_feedback_confidence = weighted_confidence
            high_confidence_spoken = False
        elif weighted_confidence > last_feedback_confidence:
            print("You're improving! Keep it up.")
            '''engine.say("You're improving! Keep it up.")
            engine.runAndWait()'''
            last_feedback_time = current_time
            last_feedback_confidence = weighted_confidence
    else:  # This covers weighted_confidence >= 0.8
        if not high_confidence_spoken:
            print("Congrats! You've got how to do this pose now.")
            '''engine.say("Congrats! You've got how to do this pose now.")
            engine.runAndWait()'''
            high_confidence_spoken = True
            last_feedback_time = current_time
            last_feedback_confidence = weighted_confidence
        elif weighted_confidence < last_feedback_confidence:
            high_confidence_spoken = False  # Reset so it can speak again if confidence goes back up
    if weighted_confidence > last_feedback_confidence:
            last_feedback_confidence = weighted_confidence

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    # Display the image
    cv2.imshow('Yoga Pose Classification', image)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
