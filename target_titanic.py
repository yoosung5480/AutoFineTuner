import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("STEP 1: LOAD TITANIC DATASET")

# CSV 파일 로드 (Kaggle에서 train.csv만 필요)
data = pd.read_csv("data/train.csv")

# 결측치 처리
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Embarked"].fillna("S", inplace=True)

# 범주형 인코딩
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 필요한 피처 선택
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data[features]
y = data["Survived"]

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train/val 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# ============================================================
# STEP 2: DEFINE BASE MODEL (Pretrained MLP)
# ============================================================

class TitanicBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        x = self.shared(x)
        x = self.head(x)
        return x

# base model (pretrained) - 가짜 pretrained라 생각
base_model = TitanicBaseModel()
pretrained_state = base_model.state_dict()  # 저장해둠

# ============================================================
# STEP 3: FINE-TUNING MODEL
# ============================================================

# 새로운 모델을 만들고, shared layer는 freeze (fine-tuning 구조)
finetune_model = TitanicBaseModel()
finetune_model.load_state_dict(pretrained_state)  # pretrained 가중치 불러오기

for param in finetune_model.shared.parameters():
    param.requires_grad = False  # shared layer freeze

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(finetune_model.head.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetune_model.to(device)

# ============================================================
# STEP 4: TRAIN (1 epoch only)
# ============================================================

print("STEP 4: TRAINING START")
num_epochs = 1
finetune_model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = finetune_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader.dataset):.4f} | Acc: {acc:.2f}%")

print("Fine-tuning completed.")
