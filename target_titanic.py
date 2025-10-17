import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# STEP 1: LOAD & PREPROCESS TITANIC DATASET
# ============================================================
print("STEP 1: LOAD TITANIC DATASET")

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

# 특징 선택
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data[features]
y = data["Survived"]

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Val 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

# ============================================================
# STEP 2: TRANSFORMER MODEL
# ============================================================

class TitanicTransformer(nn.Module):
    def __init__(self, input_dim=7, d_model=32, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # (B, input_dim) → (S=1, B, d_model)
        x = self.input_proj(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # mean pooling
        x = self.cls_head(x)
        return x


# 모델 초기화
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = TitanicTransformer().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# STEP 3: TRAIN
# ============================================================

print("STEP 3: TRAINING START")
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    train_acc = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader.dataset):.4f}, Acc: {train_acc:.2f}%")

print("Training completed.")

# ============================================================
# STEP 4: INFERENCE ON TEST SET & SAVE SUBMISSION
# ============================================================

print("STEP 4: GENERATE SUBMISSION FILE")

test_data = pd.read_csv("data/test.csv")

# 동일한 전처리 적용
test_data["Age"].fillna(data["Age"].mean(), inplace=True)
test_data["Embarked"].fillna("S", inplace=True)
for col in ["Sex", "Embarked"]:
    test_data[col] = label_encoders[col].transform(test_data[col])

X_test = test_data[features]
X_test = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# 예측
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": preds
})
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv created successfully!")
