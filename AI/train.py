import os
from PIL import Image # import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
# 학습률 리스트 설정
#learning_rates = [0.0001, 0.00001, 0.000001, 0.0000001]
learning_rates = [0.00001]
# 데이터셋 정의
class GestureDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None, img_size=64):
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.image_paths = []
        self.targets = []

        # 각 폴더에서 이미지 경로와 라벨 저장
        for label in self.labels:
            folder_path = os.path.join(data_dir, label)
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.targets.append(self.labels.index(label))  # 라벨을 인덱스로 변환

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]
        
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지 로드
        # img = cv2.resize(img, (self.img_size, self.img_size))  # 크기 조정
        img = Image.open(img_path)
        img = img.resize((self.img_size, self.img_size))
        # img = np.expand_dims(img, axis=0)  # (1, H, W) 형태로 변경

        if self.transform:
            img = self.transform(img)

        return img, label

# 데이터 경로 설정 및 라벨 정의
data_dir = "/home/koolab/create_data" # TODO: 데이터 추가할 때마다 저 경로에 데이터 새로 넣어주기 (용량이 작아도 파일 수가 많으면 오래 걸리니까 압축하기)
alphabet_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O']

# 데이터 증강 설정
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(30), # TODO: rotation 제거하기
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # TODO: crop 없이 resize만 하기
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더 생성

# 전체 데이터셋 로드
full_dataset = GestureDataset(data_dir=data_dir, labels=alphabet_labels, transform=train_transform, img_size=64)

# TODO: chatGPT한테 train, val, test 나눠달라고 하기 - check !!
# 1차 분할: 전체 데이터를 train과 test로 분리 (80% train, 20% test)
train_idx, test_idx = train_test_split(np.arange(len(full_dataset)), test_size=0.2, random_state=42)

# 2차 분할: train 데이터를 다시 train과 validation으로 분리 (90% train, 10% val)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

# Subset으로 데이터셋 생성
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)
# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# VGG 모델 정의 (사전 학습된 모델 사용)
# TODO: convolution, linear(dense), maxpool, flatten, relu, dropout layer 공부하기
model = models.vgg16(pretrained=True) # TODO: vgg16 말고 다른 모델 찾아보기 (mobilenet 같은 게 크기가 작을거임)
# model.classifier[6] = nn.Linear(4096, len(alphabet_labels))  # 마지막 레이어를 레이블 수에 맞게 수정
model.classifier = nn.Sequential(
    nn.Linear(in_features=512 * 7 * 7, out_features=1024, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1024, out_features=len(alphabet_labels), bias=True),
) # TODO: model.classifier[6]을 바꾸는 코드는 마지막 layer만 바꾸겠다고 되어 있는건데, 이거는 classifier를 전부(layer 한 3개 정도 있던 거) 바꿔버리겠다는 뜻임, 마지막 layer만도 바꿔보고 classifier 전체 구조 바꿔가면서 실험해보기
# TODO: 그런데 이렇게 하면 아래에 optimizer에는 model.classifer.parameters()를 넣어줘야 함 (하나만 바꿀 때는 model.classifier[6].parameters()만 넣으면 됨)

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001) # TODO: learning rate 바꿔가면서 실험하기, lr scheduler 찾아보기, model.classifier[6].parameters()를 model.parameters()로 바꾸면 pretrain된 부분도 다시 학습하는데, 이렇게 해서도 성능 비교하기
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5): # TODO: epoch 수 적당히 조절 (early stopping을 넣든)
    best_acc = 0.0
    epochs_no_improve = 0  # 개선되지 않은 에포크 수
    for epoch in tqdm(range(num_epochs), desc='epoch'):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='batch', leave=False):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 검증
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='test batch', leave=False):
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
        
        epoch_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc * 100:.2f}%")

        # 최상의 모델 저장
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            epochs_no_improve = 0  # 초기화
            # torch.save(model.state_dict(), 'vgg_gesture_model.pth')
            # print("최고 정확도 모델이 저장되었습니다.")
            print('빨리 돌아가게 하려고 모델 저장 주석처리 함') # TODO: 모델 저장하도록 다시 주석 지우기
        else:
            epochs_no_improve += 1
            print(f"개선되지 않은 에포크 수: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("조기 중단 발동")
                break
    return best_acc

# 여러 학습률에 대해 학습 실행
for lr in learning_rates:
    print(f"\n학습률 {lr}로 학습 시작:")
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, len(alphabet_labels))
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)
    print(f"Learning Rate: {lr}, Best Validation Accuracy: {best_accuracy * 100:.2f}%")
    
print("모든 학습이 완료되었습니다.")
# 학습된 모델 저장
torch.save(model.state_dict(), 'vgg_gesture_model_final.pth')
print("최종 모델이 저장되었습니다.")
