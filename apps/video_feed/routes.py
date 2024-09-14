import cv2
from flask import render_template, request, Response, redirect, url_for, jsonify, session
from flask_login import current_user
from apps.video_feed.dbmodels import YogaSession, HeartRateData, YogaPoseData
from datetime import datetime, timezone
from apps import db
from apps.video_feed.confident_new import extract_landmarks, predict_with_confidence, pose, model, scaler, mp_drawing, mp_pose
from apps.video_feed import blueprint
from apps.config import API_GENERATOR
from collections import deque
import time
import numpy as np
import warnings
import pyttsx3
import threading
from sqlalchemy import func

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')




def speak(text):
    def speak_thread():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    # Start the TTS in a new thread
    tts_thread = threading.Thread(target=speak_thread)
    tts_thread.start()





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

reps = 0

@blueprint.route('/get_reps')
def get_reps():
    return jsonify({'text': reps})

def update_reps(n):
    global reps
    reps = n


@blueprint.route('/save-bpm', methods=['POST'])
def save_bpm():
    bpm_data = request.json
    bpm = bpm_data['bpm']
    timestamp = bpm_data['timestamp']
    pose = bpm_data['pose_key']
    user_id=current_user.id
    
    # Assuming the session ID is already created and stored
    session_id = session.get('yoga_session_id')
    
    # Store the BPM data in the database
    heart_rate_entry = HeartRateData(
        user_id=user_id,
        session_id=session_id,
        heart_rate=bpm,
        timestamp=timestamp,
        pose_name=pose
    )
    db.session.add(heart_rate_entry)
    db.session.commit()

    return jsonify({"message": "BPM data saved successfully."})


@blueprint.route('/post_session', methods=['POST'])
def post_session():
    global session
    data = request.json
    pose_name = data["pose"]
    calories_burned = data["calories"]  # Calculate or fetch this from the session
    time_string = data["time"]  # Duration in seconds
    session_id=session.get('yoga_session_id')

    if not session_id:
        return jsonify({
            'success': False,
            'error': 'Yoga session not found'
        }), 400

    try:
        heart_rate_value = db.session.query(
        func.avg(HeartRateData.heart_rate)  # Calculate the average heart rate
        ).filter_by(
            user_id=current_user.id,
            session_id=session_id,
            pose_name=pose_name
        ).scalar()  # Use scalar() to get the single value

        heart_rate_value = round(heart_rate_value) if heart_rate_value else None
        # Round to closest integer value for better readability

        if not session_id:
                return jsonify({'success': False, 'error': 'No active session found'}), 400

        # Fetch the current YogaSession object using session_id
        yoga_session = YogaSession.query.get(session_id)
        if not yoga_session:
            return jsonify({'success': False, 'error': 'Yoga session not found'}), 404
        
        minutes, seconds = map(int, time_string.split(':'))
        time_spent = float(f"{minutes}.{seconds:02d}")
        
        # Store the pose data in YogaPoseData table
        pose_data = YogaPoseData(
            user_id=current_user.id,
            session_id=session_id,
            pose_name=pose_name,
            avg_heart_rate=heart_rate_value,
            calories_burned=calories_burned,
            time_spent=time_spent,
        )
        db.session.add(pose_data)
        db.session.commit()

        return jsonify({'success': True, 'message': 'pose data saved successfully'}), 200

    except Exception as e:
        print(f"Error saving pose data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


    

@blueprint.route('/start_session')
def start_session():
    global session

    try:
        user_id=current_user.id
        print(f"user_id: {user_id}")
        # Session creation logic
        yoga_session = YogaSession(user_id=user_id, start_time=datetime.now(timezone.utc),)
        print(f"Created session: {yoga_session}")

        # Add to session and commit
        db.session.add(yoga_session)
        db.session.commit()

        # Check if it was successfully committed
        print(f"Session ID after commit: {yoga_session.id}")

        # Store session ID in Flask session
        session['yoga_session_id'] = yoga_session.id
        print(f"Session ID stored in Flask session: {session['yoga_session_id']}")

        return jsonify({
            'success': True,
            'session_id': yoga_session.id
        }), 200
    except Exception as e:
        print(e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
led_status = "OFF"
    
# Route to fetch LED control status for ESP32 (via HTTP GET)
@blueprint.route('/led_control', methods=['GET'])
def led_control():
    global led_status
    return led_status, 200

# Route to update the LED status from the browser
@blueprint.route('/set_led/<status>', methods=['GET'])
def set_led(status):
    global led_status
    if status in ["ON", "OFF"]:
        led_status = status
        print(f"LED status set to: {led_status}")
        return jsonify({"message": f"LED status set to {led_status}"}), 200
    return jsonify({"error": "Invalid status"}), 400


@blueprint.route('/interface/<int:pose_index>')
def analyze(pose_index):
    global fsrentry
    pose_data = [
    {"name": "Tree Pose (Vrikshasana)", "level": "Beginner", "pose_key": "Tree"},
    {"name": "Chair Pose (Bhujangasana)", "level": "Beginner", "pose_key": "Chair"},
    {"name": "Cobra Pose (Utkatasana)", "level": "Intermediate", "pose_key": "Cobra"},
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
        is_last_pose=is_last_pose
        )



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
CONFIDENCE_THRESHOLD = 0.7  # Increased for more reliable predictions

# Pose hold time threshold
POSE_HOLD_THRESHOLD = 5  # Increased to 5 seconds for a more challenging hold

# Last audio feedback time
last_audio_time = 0
AUDIO_COOLDOWN = 5  # Cooldown period for audio feedback in seconds

# Weighted confidence variables
confidence_window = deque(maxlen=10)  # Store the last 10 confidence values

noProgress = 0
# Last detected pose
last_detected_pose = None

cap = cv2.VideoCapture(1)

def generate_frames(target):
    global camera_on, cap, noProgress, smoothed_landmarks, last_audio_time, last_detected_pose, last_feedback_confidence, last_feedback_time, high_confidence_spoken
    global pose_tracker, current_pose, pose_hold_time, pose_start_time, rep_count, alpha, CONFIDENCE_THRESHOLD, POSE_HOLD_THRESHOLD, AUDIO_COOLDOWN
    cap = cv2.VideoCapture(1)
    camera_on = True
    if not target:
        update_text("no target")
    else:
        update_text(f"target detected: {target}")
    
    speak(yoga_poses_data[target])
    

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

        

        confidence = 0
        weighted_confidence = 0
        
        if landmarks is not None:
            # Apply smoothing filter
            if smoothed_landmarks is None:
                smoothed_landmarks = landmarks
            else:
                smoothed_landmarks = alpha * landmarks + (1 - alpha) * smoothed_landmarks

        
            
            # Make prediction
            prediction, confidence = predict_with_confidence(model, scaler, landmarks)

             # Update confidence window
            confidence_window.append(confidence)
            
            # Calculate weighted confidence
            weighted_confidence = np.average(confidence_window, weights=range(1, len(confidence_window) + 1))
            

            # Update pose tracker
            if weighted_confidence > CONFIDENCE_THRESHOLD:
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
                        cv2.putText(image, f"Confidence: {weighted_confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        

                        # Timer for pose hold
                        current_time = time.time()
                        if current_pose in yoga_poses_data and weighted_confidence > CONFIDENCE_THRESHOLD:
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
                            update_reps(rep_count)
                        cv2.putText(image, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                        
                    else:
                        update_text(f"Detected pose '{current_pose}' does not match target '{target}'. Ignoring...")
                        cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        current_time = time.time()
        

        
        if weighted_confidence <= 0.5:
            if current_time - last_feedback_time >= 5:  # Check if 3 seconds have passed
                
                noProgress+=1
                
                if noProgress == 20:
                    speak(yoga_poses_data[target])
                    update_text("pose not right 20 times")
                else:
                    update_text(f"Pose not right. Try again Please. {noProgress}")
                
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
                high_confidence_spoken = False
            elif weighted_confidence > last_feedback_confidence:
                update_text("That's better! Keep improving.")
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
        elif 0.5 < weighted_confidence < 0.8:
            if current_time - last_feedback_time >= 5:  # Check if 3 seconds have passed
                update_text("Good going. Keep trying!")
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
                high_confidence_spoken = False
            elif weighted_confidence > last_feedback_confidence:
                update_text("You're improving! Keep it up.")
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
        else:  # This covers confidence >= 0.8
            if not high_confidence_spoken:
                update_text("Congrats! You've got how to do this pose now.")
                speak("Congrats! You've got how to do this pose now.")
                high_confidence_spoken = True
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
            elif weighted_confidence < last_feedback_confidence:
                high_confidence_spoken = False  # Reset so it can update_text again if confidence goes back up
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
    global camera_on, cap, smoothed_landmarks, last_audio_time, last_detected_pose, last_feedback_confidence, last_feedback_time, high_confidence_spoken
    global pose_tracker, current_pose, pose_hold_time, pose_start_time, rep_count, alpha, CONFIDENCE_THRESHOLD, POSE_HOLD_THRESHOLD, AUDIO_COOLDOWN
    global reps, current_text, session
    
    
    print("initialized")
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
    CONFIDENCE_THRESHOLD = 0.7  # Increased for more reliable predictions

    # Pose hold time threshold
    POSE_HOLD_THRESHOLD = 5  # Increased to 5 seconds for a more challenging hold

    # Last audio feedback time
    last_audio_time = 0
    AUDIO_COOLDOWN = 5  # Cooldown period for audio feedback in seconds

    # Weighted confidence variables
    confidence_window = deque(maxlen=10)  # Store the last 10 confidence values
    reps=0
    current_text = ''

    # Last detected pose
    last_detected_pose = None
    camera_on = False

     # Fetch current session ID from the Flask session
    session_id = session.get('yoga_session_id')
    print(session_id)

    if session_id:
        yoga_session = YogaSession.query.get(session_id)
        if yoga_session:
            # Set the end time of the session
            yoga_session.end_time = datetime.now(timezone.utc)

            # Calculate the total duration and total calories
            yoga_session.calculate_total_duration()
            yoga_session.calculate_total_calories()

            # Commit the updated session data to the database
            db.session.commit()

            # Clear the session ID after session ends
            session.pop('yoga_session_id', None)

    cap.release()
    try:
        return redirect(url_for('home_blueprint.index'))
    except Exception as e:
        print(f"error while redirecting to home: {e}")