import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 학습된 모델 불러오기
with open("voting_cross_validated.p", "rb") as f:
    model = pickle.load(f)

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 한글 폰트 설정
font_path = "malgun.ttf"  # 폰트 파일 경로
font = ImageFont.truetype(font_path, 30)

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

    # 데이터 초기화
    left_hand_data = [0] * 42  # 왼손 데이터 초기화
    right_hand_data = [0] * 42  # 오른손 데이터 초기화

    if results.multi_hand_landmarks and results.multi_handedness:
        # 감지된 손의 데이터를 분리
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = hand_info.classification[0].label  # "Left" 또는 "Right"
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x)
                hand_data.append(landmark.y)
            if handedness == "Left":
                left_hand_data = hand_data
            elif handedness == "Right":
                right_hand_data = hand_data

        # 두 손 데이터를 결합하여 하나의 입력 벡터로 생성
        combined_data = np.array(left_hand_data + right_hand_data).reshape(1, -1)

        # 모델 예측
        prediction = model.predict(combined_data)[0]

        # OpenCV 이미지에 Pillow로 텍스트 추가
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 50), f'예측 결과: {prediction}', font=font, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 손 랜드마크 그리기
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 결과 화면 표시
    cv2.imshow('Real-time Gesture Prediction', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
