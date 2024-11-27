import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
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

# 데이터 수집 상태 플래그 및 타이머
collecting = False
start_time = None

def collect_data():
    global collecting, start_time
    collecting = True
    start_time = time.time()  # 시작 시간 기록

print("데이터 수집을 시작합니다. 's' 키를 눌러 데이터를 수집하세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # BGR 이미지를 RGB로 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Mediapipe로 손 랜드마크 감지
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        # 초기화: 왼손, 오른손 데이터를 각각 42개씩 확보 (84개)
        left_hand_data = [0] * 42
        right_hand_data = [0] * 42

        for hand_idx, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # 손잡이 정보 가져오기
            handedness = hand_info.classification[0].label  # "Left" 또는 "Right"

            # 랜드마크 데이터 저장
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x)
                hand_data.append(landmark.y)

            # 왼손 또는 오른손에 따라 데이터 저장
            if handedness == "Left":
                left_hand_data = hand_data
            elif handedness == "Right":
                right_hand_data = hand_data

        # 두 손 데이터를 병합하여 하나의 데이터 포인트로 저장
        data_point = left_hand_data + right_hand_data

        # 데이터 수집 조건
        if collecting:
            elapsed_time = time.time() - start_time

            if elapsed_time >= 3:  # 3초 대기 후 수집 시작
                if len(data) == 0 or elapsed_time >= 3 + (len(data) * 0.3):
                    data.append(data_point)
                    labels.append(gesture_label)
                    print(f"제스처 '{gesture_label}' 데이터 수집 중... 총 {len(data)}개")

                # 100개 데이터가 수집되면 종료
                if len(data) >= 100:
                    print("100개의 데이터가 수집되었습니다. 프로그램을 종료합니다.")
                    collecting = False
                    break
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    # 결과 화면 표시
    cv2.imshow('Gesture Data Collection', frame)

    # 's' 키를 누르면 데이터 수집 시작
    if cv2.waitKey(1) & 0xFF == ord('s') and not collecting:
        collect_data()

# 새로운 데이터를 기존 데이터에 추가하여 저장
with open(file_path, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)
print(f"제스처 '{gesture_label}' 데이터가 '{gesture_label}.pickle'에 추가 저장되었습니다. 총 데이터 수: {len(data)}개")

# 자원 해제 
cap.release()
cv2.destroyAllWindows()
 