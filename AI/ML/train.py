import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os

# 데이터 불러오기
data_dir = "./gesture_data"
data = []
labels = []

# 라벨 변경 함수
def transform_label(label):
    if label == "AAAA":
        return "A"
    elif label == "gG":
        return "T"
    else:
        return label  # 다른 라벨은 그대로 유지
    
for file in os.listdir(data_dir):
    if file.endswith(".pickle"):
        with open(os.path.join(data_dir, file), "rb") as f:
            gesture_data = pickle.load(f)
            data.extend(gesture_data["data"])
            # 라벨 변경 적용
            labels.extend([transform_label(lbl) for lbl in gesture_data["labels"]])

# 데이터 배열 변환
X = np.array(data)
y = np.array(labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# 개별 모델 정의
rf_model = RandomForestClassifier(random_state=22)
dt_model = DecisionTreeClassifier(random_state=22)
knn_model = KNeighborsClassifier()
svm_model = SVC(probability=True, random_state=22)

# VotingClassifier 정의
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('dt', dt_model), ('svm', svm_model), ('knn', knn_model)], voting='soft'
)

# 교차 검증
cv_scores = cross_val_score(voting_model, X, y, cv=5)  # 5-폴드 교차 검증
print(f"Voting 앙상블 교차 검증 정확도: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# VotingClassifier 학습
voting_model.fit(X_train, y_train)



# 학습된 모델 저장
with open("voting_cross_validated.p", "wb") as f:
    pickle.dump(voting_model, f)
print("교차 검증을 거친 Voting 앙상블 모델이 'voting_cross_validated.json'에 저장되었습니다.")
