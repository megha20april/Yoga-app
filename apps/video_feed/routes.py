import cv2
from flask import render_template, request, Response, redirect, url_for, jsonify
from apps.video_feed.confident_new import extract_landmarks, predict_with_confidence, pose, model, scaler, mp_drawing, mp_pose
from apps.video_feed import blueprint
from apps.config import API_GENERATOR
from collections import deque
import time
import numpy as np
import warnings
import pyttsx3
import threading
import queue

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech


# Queue for managing TTS requests
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

threading.Thread(target=tts_worker).start()

def speak(text):
    tts_queue.put(text)

    # Run the speak function in a new thread

camera_on = False

# List of yoga poses and their instructions
yoga_poses_data = {
    "Chair": "Stand with your feet together, then bend your knees and lower your hips as if sitting in a chair. Raise your arms overhead.",
    "Cobra": "Lie face down, place your hands under your shoulders, then lift your chest while keeping your hips on the ground.",
    "Tree": "Stand on one leg, place the sole of your other foot on your inner thigh or calf, and bring your hands together at your chest.",
}

current_text = ""

@blueprint.route('/get_text')
def get_text():
    return jsonify({'text': current_text})

def update_text(text):
    global current_text
    current_text = text



@blueprint.route('/interface/<int:pose_index>')
def analyze(pose_index):
    pose_data = [
    {"name": "Cobra Pose (Utkatasana)", "level": "Beginner", "pose_key": "Cobra"},
    {"name": "Chair Pose (Bhujangasana)", "level": "Beginner", "pose_key": "Chair"},
    {"name": "Tree Pose (Vrikshasana)", "level": "Intermediate", "pose_key": "Tree"},
    # Add more poses as needed
    ]

    if pose_index >= len(pose_data):
        return redirect(url_for('home_blueprint.index'))
    
    pose = pose_data[pose_index]
    next_pose_index = pose_index + 1
    is_last_pose = next_pose_index >= len(pose_data)

    return render_template(
        'home/interface.html',
        segment='interface', 
        API_GENERATOR=len(API_GENERATOR), 
        show_sideBar=False,
        show_nav=False,
        pose=pose, 
        next_pose_index=next_pose_index,
        is_last_pose=is_last_pose)



#initializing variables for feedback mech
last_feedback_time = 0
last_feedback_confidence = 0
high_confidence_spoken = False

# Pose tracking
pose_tracker = deque(maxlen=15)
current_pose = None
pose_start_time = None
pose_hold_time = 0
rep_count = 0

# Smoothing filter
alpha = 0.3  # Adjusted for more responsive smoothing
smoothed_landmarks = None

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.8  # Increased for more reliable predictions

# Pose hold time threshold
POSE_HOLD_THRESHOLD = 5  # Increased to 5 seconds for a more challenging hold

# Last audio feedback time
last_audio_time = 0
AUDIO_COOLDOWN = 5  # Cooldown period for audio feedback in seconds

# Last detected pose
last_detected_pose = None

cap = cv2.VideoCapture(1)

def generate_frames(target):
    global camera_on, smoothed_landmarks, last_audio_time, last_detected_pose, last_feedback_confidence, last_feedback_time, high_confidence_spoken
    global pose_tracker, current_pose, pose_hold_time, pose_start_time, rep_count, alpha, CONFIDENCE_THRESHOLD, POSE_HOLD_THRESHOLD, AUDIO_COOLDOWN
    cap = cv2.VideoCapture(1)
    camera_on = True
    if not target:
        update_text("no target")
    else:
        update_text(f"target detected: {target}")
    
    
    

    while camera_on:
        ret, frame = cap.read()
        if not ret: 
            update_text("Camera feed failed. Restarting the camera.")
            cap.release()
            cap = cv2.VideoCapture(1)  # Re-initialize the camera
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get the landmarks
        results = pose.process(image)
        
        # Convert the image back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks = extract_landmarks(results)

        speak(yoga_poses_data[target])
        
        if landmarks is not None:
            # Apply smoothing filter
            if smoothed_landmarks is None:
                smoothed_landmarks = landmarks
            else:
                smoothed_landmarks = alpha * landmarks + (1 - alpha) * smoothed_landmarks

        
            global prediction, confidence
            # Make prediction
            prediction, confidence = predict_with_confidence(model, scaler, landmarks)
        

            # Update pose tracker
            if confidence > CONFIDENCE_THRESHOLD:
                pose_tracker.append(prediction)


                if len(pose_tracker) == pose_tracker.maxlen:
                    current_pose = max(set(pose_tracker), key=pose_tracker.count)
                    # Map the detected pose to one of the seven yoga poses or handle 'no_pose'
                    matched_pose = False
                    for yoga_pose in yoga_poses_data.keys():
                        if current_pose.lower() in yoga_pose.lower():
                            current_pose = yoga_pose
                            matched_pose = True
                            break
                    if not matched_pose:
                        current_pose = "Unknown Pose"
                    
                    # Proceed only if the detected pose matches the target pose
                    if target.lower() in current_pose.lower():
                        update_text(f"Target pose detected: {current_pose}")
                        last_detected_pose = current_pose

                        
                        # Display the prediction and confidence
                        cv2.putText(image, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        

                        # Timer for pose hold
                        current_time = time.time()
                        if current_pose in yoga_poses_data and confidence > CONFIDENCE_THRESHOLD:
                            if pose_start_time is None:
                                pose_start_time = current_time
                            pose_hold_time = current_time - pose_start_time
                            cv2.putText(image, f"Hold Time: {pose_hold_time:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                        # Audio feedback only for known poses
                        if current_time - last_audio_time > AUDIO_COOLDOWN:
                            if pose_hold_time < POSE_HOLD_THRESHOLD:
                                update_text(f"Continue holding the {current_pose} for {POSE_HOLD_THRESHOLD - pose_hold_time:.0f} more seconds")
                            elif pose_hold_time >= POSE_HOLD_THRESHOLD:
                                update_text(f"Excellent! You've held the {current_pose} successfully")
                            last_audio_time = current_time

                        # Rep counter
                        if current_pose in yoga_poses_data and pose_hold_time > POSE_HOLD_THRESHOLD:
                            rep_count += 1
                            pose_start_time = None  # Reset timer
                            update_text(f"Rep {rep_count} completed")
                        cv2.putText(image, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                        
                    else:
                        update_text(f"Detected pose '{current_pose}' does not match target '{target}'. Ignoring...")
                        cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        current_time = time.time()

        if confidence <= 0.5:
            if current_time - last_feedback_time >= 5:  # Check if 3 seconds have passed
                update_text("Pose not right. Try again Please.")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
                high_confidence_spoken = False
            elif confidence > last_feedback_confidence:
                update_text("That's better! Keep improving.")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
        elif 0.5 < confidence < 0.8:
            if current_time - last_feedback_time >= 5:  # Check if 3 seconds have passed
                update_text("Good going. Keep trying!")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
                high_confidence_spoken = False
            elif confidence > last_feedback_confidence:
                update_text("You're improving! Keep it up.")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
        else:  # This covers confidence >= 0.8
            if not high_confidence_spoken:
                update_text("Congrats! You've got how to do this pose now.")
                high_confidence_spoken = True
                last_feedback_time = current_time
                last_feedback_confidence = confidence
            elif confidence < last_feedback_confidence:
                high_confidence_spoken = False  # Reset so it can update_text again if confidence goes back up
        if confidence > last_feedback_confidence:
                last_feedback_confidence = confidence


        
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
        (flag, buffer) = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Break the loop when 'q' is pressed
        yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()
    camera_on = False


@blueprint.route('/video_feed/<target>')
def video_feed(target):
    
    return Response(generate_frames(target), mimetype='multipart/x-mixed-replace; boundary=frame')

@blueprint.route('/stop_video')
def stop_video():
    global camera_on, cap
    camera_on = False
    cap.release()
    return redirect(url_for('home_blueprint.index'))