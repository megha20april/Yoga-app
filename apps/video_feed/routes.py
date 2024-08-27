import cv2
from flask import render_template, request, Response, redirect, url_for
from apps.video_feed.confident_new import extract_landmarks, predict_with_confidence, pose, model, scaler, mp_drawing, mp_pose
from apps.video_feed import blueprint
from apps.config import API_GENERATOR

camera_on = False



@blueprint.route('/interface/<int:pose_index>')
def analyze(pose_index):
    pose_data = [
    {"name": "Cobra Pose (Bhujangasana)", "level": "Beginner", "pose_key": "Cobra"},
    {"name": "Chair Pose (...)", "level": "Beginner", "pose_key": "Chair"},
    {"name": "Tree Pose (Vrksasana)", "level": "Intermediate", "pose_key": "Tree"},
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
        pose=pose, 
        next_pose_index=next_pose_index,
        is_last_pose=is_last_pose)


def generate_frames():
    global camera_on
    camera_on = True
    cap = cv2.VideoCapture(0)
    while camera_on:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and get the landmarks
        results = pose.process(image)
        
        # Convert the image back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks = extract_landmarks(results)
        
        if landmarks is not None:
            # Make prediction
            prediction, confidence = predict_with_confidence(model, scaler, landmarks)
            
            # Display prediction and confidence
            cv2.putText(image, f"Pose: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the image
        (flag, buffer) = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Break the loop when 'q' is pressed
        yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()


@blueprint.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@blueprint.route('/stop_video')
def stop_video():
    global camera_on
    camera_on = False
    return redirect(url_for('home_blueprint.index'))