import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

# Load the trained model
with open("voting_cross_validated.p", "rb") as f:
    model = pickle.load(f)

# Mediapipe settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

# Variables for gesture recognition and sentence creation
current_alphabet = None
last_detected_time = None
sentence = ""
last_added_time = None
reset_time = None
cursor_visible = True
cursor_last_toggle = time.time()

print("Starting real-time gesture prediction. Press 'q' to exit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the camera.")
        break

    # Convert the BGR image to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks using MediaPipe
    results = hands.process(img_rgb)

    # Current time
    now = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark data
            data_point = []
            for landmark in hand_landmarks.landmark:
                data_point.append(landmark.x)
                data_point.append(landmark.y)

            # Model prediction
            data_point = np.array(data_point).reshape(1, -1)
            try:
                prediction = model.predict(data_point)[0]
            except Exception as e:
                print(f"Prediction error: {e}")
                continue
            # Update current alphabet and detection time
            if prediction == current_alphabet:
                if now - last_detected_time > 2:  # Add alphabet to the sentence if held for 2 seconds
                    if last_added_time is None or now - last_added_time > 2:
                        sentence += prediction
                        last_added_time = now
                        dot_added_time = None  # Reset '.' timer after adding an alphabet
            else:
                current_alphabet = prediction
                last_detected_time = now

            reset_time = now  # Reset idle time tracker
            
            # Display the current prediction on the bottom right
            cv2.putText(frame, f'Current: {current_alphabet}', (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    else:
        # No detection logic
        if reset_time and now - reset_time > 2:
            if dot_added_time is None:  # Add '.' only once after 2 seconds of no detection
                sentence += "."
                dot_added_time = now  # Track when '.' was added
            elif now - dot_added_time > 2:  # Clear sentence 2 seconds after adding '.'
                sentence = ""
                reset_time = None
                dot_added_time = None
                current_alphabet = None
                
    # Toggle cursor visibility every 0.5 seconds
    if now - cursor_last_toggle > 0.5:
        cursor_visible = not cursor_visible
        cursor_last_toggle = now

    # Display the assembled sentence with a blinking cursor
    cursor = "|" if cursor_visible else " "
    display_text = sentence + cursor
    cv2.putText(frame, f'Sentence: {display_text}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the current prediction on the bottom right
    if current_alphabet:
        cv2.putText(frame, f'Current: {current_alphabet}', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Gesture Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
