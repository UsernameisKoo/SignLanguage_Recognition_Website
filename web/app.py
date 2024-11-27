from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
from gpt import process_data

app = Flask(__name__)

# 머신러닝 모델 로드
with open("voting_cross_validated.p", "rb") as f:
    model = pickle.load(f)

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 카메라 초기화 변수
camera_active = False
cap = None

# 한글 폰트 설정
font_path = "malgun.ttf"  # Windows 환경에서 사용할 폰트 파일 경로
font = ImageFont.truetype(font_path, 30)

# 제스처 인식 변수
current_alphabet = None
last_alphabet = None
last_detected_time = 0
sentence = ""

def process_hand_data(hand_landmarks):
    """Extracts hand data for prediction."""
    hand_data = []
    for landmark in hand_landmarks.landmark:
        hand_data.append(landmark.x)
        hand_data.append(landmark.y)
    return hand_data

def generate_frames():
    global current_alphabet, last_alphabet, last_detected_time, sentence, cap
    if cap is None or not cap.isOpened():
        print("Error: Camera not initialized or not opened.")
        return

    while camera_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # BGR -> RGB 변환
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe로 손 랜드마크 감지
        results = hands.process(img_rgb)

        # 알파벳 초기화
        current_alphabet = None

        # Initialize hand data
        left_hand_data = [0] * 42
        right_hand_data = [0] * 42
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = hand_info.classification[0].label  # "Left" or "Right"
                if handedness == "Left":
                    left_hand_data = process_hand_data(hand_landmarks)
                elif handedness == "Right":
                    right_hand_data = process_hand_data(hand_landmarks)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Combine left and right hand data
            combined_data = np.array(left_hand_data + right_hand_data).reshape(1, -1)

            # Model prediction
            try:
                prediction = model.predict(combined_data)[0]
                current_alphabet = prediction
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

            # 알파벳 업데이트 및 시간 체크
            if current_alphabet:
                now = time.time()
                if current_alphabet == last_alphabet:
                    if now - last_detected_time >= 1.5:  # 1.5초 이상 유지
                        if current_alphabet.isascii():
                            sentence += current_alphabet
                        else:
                            sentence += current_alphabet
                            sentence += " "
                        last_detected_time = now
                else:
                    last_alphabet = current_alphabet
                    last_detected_time = now

        # 화면에 현재 알파벳 및 문장 표시 (PIL로 한글 지원)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # 한글 텍스트 표시
        if current_alphabet:
            draw.text((10, 30), f'현재: {current_alphabet}', font=font, fill=(0, 255, 0))

        # PIL 이미지를 다시 OpenCV 형식으로 변환
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 프레임 반환
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
@app.route('/')
def index():
    global sentence
    sentence = ""  # 새로고침 시 sentence 초기화
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
    sentence = sentence.rstrip() + "."
    return jsonify({"sentence": sentence})

@app.route('/translate_sentence', methods=['POST'])
def translate_sentence():
    global sentence
    translated_sentence = process_data(sentence)  # 번역 결과
    return jsonify({"sentence": translated_sentence})  # 번역된 결과만 반환

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
    app.run(host='0.0.0.0', port=5000, debug=True)





# http://127.0.0.1:5000/video_feed
# "C:/Users/user/Projects/ASL/web/voting_cross_validated.pkl"

