from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import os

app = Flask(__name__, template_folder=os.path.abspath(os.getcwd()))

print("Flask working directory:", os.getcwd())
print("Looking for templates in:", os.path.abspath("docs"))


print("Loading model...")
with open("voting_cross_validated.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully.")

# Mediapipe 설정
print("Initializing Mediapipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
print("Mediapipe initialized successfully.")
mp_drawing = mp.solutions.drawing_utils

# 카메라 초기화 변수
camera_active = False
cap = None

# 제스처 인식 변수
current_alphabet = None
last_alphabet = None
last_detected_time = 0
sentence = ""

def generate_frames():
    global current_alphabet, last_alphabet, last_detected_time, sentence, cap
    if cap is None or not cap.isOpened():
        print("Error: Camera not initialized or not opened.")  # 디버깅 메시지
        return

    while camera_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")  # 디버깅 메시지
            break

        # BGR -> RGB 변환
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe로 손 랜드마크 감지
        results = hands.process(img_rgb)

        # 알파벳 초기화
        current_alphabet = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 데이터 추출
                data_point = []
                for landmark in hand_landmarks.landmark:
                    data_point.append(landmark.x)
                    data_point.append(landmark.y)

                # 모델 예측
                data_point = np.array(data_point).reshape(1, -1)
                try:
                    prediction = model.predict(data_point)[0]
                    current_alphabet = prediction
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue

                # 알파벳 업데이트 및 시간 체크
                now = time.time()
                if current_alphabet == last_alphabet:
                    # 같은 알파벳이 일정 시간 이상 유지되면 추가
                    if now - last_detected_time >= 2:  # 1초 이상 유지
                        sentence += current_alphabet
                        last_detected_time = now
                else:
                    # 알파벳 변경 시 초기화
                    last_alphabet = current_alphabet
                    last_detected_time = now

                # 랜드마크 표시
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 화면에 현재 알파벳 표시
        if current_alphabet:
            cv2.putText(frame, f'Current: {current_alphabet}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 프레임 반환
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")  # 인코딩 실패
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    

@app.route('/')
def index():
    # Render index.html and pass the sentence variable
    return render_template('index.html', sentence=sentence)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, camera_active
    if cap is None:
        for i in range(5):  # 최대 5개의 장치 번호를 테스트
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera {i} opened successfully.")
                break
            else:
                cap.release()
                cap = None  # 장치 초기화 실패 시 None으로 설정

    if cap is None or not cap.isOpened():
        print("Error: Camera initialization failed.")  # 디버깅 메시지 추가
        return jsonify({"status": "Camera initialization failed"})

    # 성공적으로 카메라 초기화
    camera_active = True
    print("Camera started successfully.")  # 디버깅 메시지
    return jsonify({"status": "Camera started"})


@app.route('/complete_sentence', methods=['POST'])
def complete_sentence():
    global sentence
    sentence += "."
    return jsonify({"status": "Sentence completed", "sentence": sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = ""
    return jsonify({"status": "Sentence cleared"})

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    global sentence
    return jsonify({"sentence": sentence})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)


