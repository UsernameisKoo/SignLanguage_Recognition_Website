import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 학습된 모델 불러오기
with open("voting_cross_validated_gesture_model.p", "rb") as f: # 교차 검증 & 보팅 모델
#with open("cross_validated_gesture_model.p", "rb") as f: # 교차 검증 모델
#with open("gesture_model.p", "rb") as f: # only 결정트리 모델
    model = pickle.load(f)

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 한글 폰트 설정 (예: 나눔고딕)
font_path = r"C:\Users\user\Projects\korean\font\NanumGothic_Regular.ttf"  # 시스템에 설치된 한글 폰트 경로
font = ImageFont.truetype(font_path, 32)

# 카메라 열기
cap = cv2.VideoCapture(0)

print("실시간 제스처 예측을 시작합니다. 'q' 키를 눌러 종료하세요.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # BGR 이미지를 RGB로 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe로 손 랜드마크 감지
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 좌표 추출
            data_point = []
            for landmark in hand_landmarks.landmark:
                data_point.append(landmark.x)
                data_point.append(landmark.y)

            # 모델 예측
            data_point = np.array(data_point).reshape(1, -1)
            prediction = model.predict(data_point)[0]
            probabilities = model.predict_proba(data_point)[0]  # 각 클래스에 대한 확률
            confidence = np.max(probabilities) * 100  # 최고 확률

            # OpenCV 이미지를 PIL 이미지로 변환
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # 한글 텍스트 출력
            if(confidence>=60):
                text = f'Prediction: {prediction} ({confidence:.2f}%)'
                draw.text((10, 50), text, font=font, fill=(0, 255, 0, 0))

            # PIL 이미지를 다시 OpenCV 이미지로 변환
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 결과 화면 표시
    cv2.imshow('Real-time Gesture Prediction', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()