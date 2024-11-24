import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 제스처 레이블 설정
gesture_label = input("저장할 제스처 레이블을 입력하세요: ")

# 데이터 저장 디렉토리 설정
data_dir = "./gesture_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 저장할 파일 경로 설정
file_path = os.path.join(data_dir, f"{gesture_label}.pickle")

# 기존 데이터 불러오기 (파일이 존재하면)
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        saved_data = pickle.load(f)
        data = saved_data["data"]
        labels = saved_data["labels"]
    print(f"기존에 저장된 {len(data)}개의 데이터가 로드되었습니다.")
else:
    data = []
    labels = []

# 카메라 열기
cap = cv2.VideoCapture(0)

print("데이터 수집을 시작합니다. 's' 키를 눌러 데이터를 수집하고, 'q' 키를 눌러 종료하세요.")
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
            
            # 's' 키를 누르면 데이터 수집
            if cv2.waitKey(1) & 0xFF == ord('s'):
                data.append(data_point)
                labels.append(gesture_label)
                print(f"제스처 '{gesture_label}' 데이터 수집 중... 총 {len(data)}개")

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 결과 화면 표시
    cv2.imshow('Gesture Data Collection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 새로운 데이터를 기존 데이터에 추가하여 저장
with open(file_path, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)
print(f"제스처 '{gesture_label}' 데이터가 '{gesture_label}.pickle'에 추가 저장되었습니다. 총 데이터 수: {len(data)}개")

# 자원 해제 
cap.release()
cv2.destroyAllWindows()
