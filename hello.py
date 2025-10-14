import sys
import json
import time
import pickle
import os
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class _Tee:
    def __init__(self, log_file_path: Path):
        self._terminal = sys.stdout
        self._log = open(log_file_path, "w", encoding="utf-8")

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        try:
            self._log.close()
        except Exception:
            pass


def _save_model_generic(model, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch as _torch
        _torch.save(getattr(model, "state_dict", lambda: model)(), path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return str(path)


def _emit_json_line(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


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


def autofinetuner_result() -> dict:
    import argparse

    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="save_path")
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--healthcheck", action="store_true")
    # Additional hyperparameters from original code
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test_size", type=float, default=0.2)

    args, _ = parser.parse_known_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.save_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"{timestamp}.log"
    tee = _Tee(log_path)
    sys.stdout = tee

    print("STEP 1: LOAD TITANIC DATASET")

    # (1) 데이터 로드 및 전처리
    data = pd.read_csv(args.train_path)

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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # 텐서 변환
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # base model (pretrained) - 가짜 pretrained라 생각
    base_model = TitanicBaseModel()
    pretrained_state = base_model.state_dict()  # 저장해둠

    # (2) 모델 구성 및 학습 설정
    finetune_model = TitanicBaseModel()
    finetune_model.load_state_dict(pretrained_state)  # pretrained 가중치 불러오기

    for param in finetune_model.shared.parameters():
        param.requires_grad = False  # shared layer freeze

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(finetune_model.head.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_model.to(device)

    # (3) TRAIN
    print("STEP 4: TRAINING START")
    num_epochs = args.epochs
    finetune_model.train()

    last_train_acc = 0.0
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

        epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0
        last_train_acc = acc
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")

    print("Fine-tuning completed.")

    # (4) 검증 및 평가지표 계산
    finetune_model.eval()
    val_correct, val_total = 0, 0
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = finetune_model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_acc_percent = 100.0 * val_correct / val_total if val_total > 0 else 0.0
    val_loss = val_running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0

    # (5) 모델 저장 (<save_dir>/model.pt)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(save_dir) / "model.pt"
    model_pt_path = _save_model_generic(finetune_model, model_path)

    # (6) result.json 파일 작성 (스켈레톤 기반)
    start_time_str = datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - t0

    params_payload = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "test_size": args.test_size,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "device": str(device),
    }

    # json skeleton keyed by timestamp (실행시간)
    result = {
        timestamp: {
            "validation_score": val_acc_percent,
            "train_score": last_train_acc,
            "loss": {
                "train_loss_last_epoch": epoch_loss if num_epochs > 0 else None,
                "val_loss": val_loss,
            },
            "runtime_info": {
                "start_time": start_time_str,
                "end_time": end_time_str,
                "elapsed_time_sec": elapsed,
            },
            "data": {
                "features": features,
                "n_train": int(len(train_dataset)),
                "n_val": int(len(val_dataset)),
                "train_path": args.train_path,
                "test_path": args.test_path,
            },
            "artifacts": {
                "model_pt_path": model_pt_path,
                "save_dir": str(save_dir),
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "result_json_path": str(run_dir / "result.json"),
            },
            "params": params_payload,
        },
        # API 계약 필드 (기존 유지)
        "model_pt_path": model_pt_path,
        "validation": val_acc_percent,
        "params": params_payload,
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # (7) stdout에 JSON 한 줄 출력 및 로그 반영
    _emit_json_line(result)

    # do not close tee before returning to ensure final flush
    tee.flush()
    return result


if __name__ == "__main__":
    print("HELLO WORLD")

