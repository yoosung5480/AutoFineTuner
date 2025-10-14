import sys
import os
import json
import time
import datetime
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _save_model_generic(model, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, path)
    except Exception:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return str(path)

def _emit_json_line(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def autofinetuner_result() -> dict:
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--healthcheck", type=int, choices=[0,1], default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    args, _ = parser.parse_known_args()
    hc = bool(args.healthcheck)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(args.save_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{timestamp}.log"
    sys.stdout = open(log_path, "w", encoding="utf-8")
    sys.stderr = sys.stdout

    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))

    try:
        print("STEP 1: LOAD TITANIC DATASET")
        data = pd.read_csv(args.train_path)
        data["Age"].fillna(data["Age"].mean(), inplace=True)
        data["Embarked"].fillna("S", inplace=True)
        label_encoders = {}
        for col in ["Sex", "Embarked"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        X = data[features]
        y = data["Survived"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # test_size 사용자 인자로
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=args.test_size, random_state=42
        )
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.long)

        # 배치사이즈 사용자 인자로
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

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

        print("STEP 2: BASE MODEL (PRETRAINED)")
        base_model = TitanicBaseModel()
        pretrained_state = base_model.state_dict()

        print("STEP 3: FINETUNE MODEL LOADING (FREEZE SHARED)")
        finetune_model = TitanicBaseModel()
        finetune_model.load_state_dict(pretrained_state)
        for param in finetune_model.shared.parameters():
            param.requires_grad = False
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(finetune_model.head.parameters(), lr=args.lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        finetune_model.to(device)

        if hc:
            print("READY")
            sys.stdout.close()
            return {"healthcheck": 1}

        print("STEP 4: TRAINING START")
        num_epochs = args.epochs
        best_train_correct, best_val_correct = 0, 0
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
            acc = 100.0 * correct / total if total else 0.0
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader.dataset):.4f} | Acc: {acc:.2f}%")

        print("STEP 5: EVALUATION (TRAIN & VAL)")
        def eval_accuracy(model, loader):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    total += labels.size(0)
                    correct += preds.eq(labels).sum().item()
            return correct / total if total else 0.0
        train_score = eval_accuracy(finetune_model, train_loader)
        val_score = eval_accuracy(finetune_model, val_loader)
        print(f"Train acc: {train_score:.4f}, Val acc: {val_score:.4f}")

        print("STEP 6: SAVING MODEL")
        model_path = run_dir / "model.pt"
        _save_model_generic(finetune_model, model_path)
        end_time = time.time()
        result = {
            timestamp: {
                "validation_score": val_score,
                "train_score": train_score,
                "params": {
                    "test_size": args.test_size,
                    "num_epochs": num_epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "save_path": str(run_dir),
                    "healthcheack": args.healthcheck
                },
                "runtime_info": {
                    "start_time": start_time_str,
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                    "elapsed_time_sec": round(end_time - t0, 4)
                },
                "execution_status": {
                    "success": True,
                    "error_type": None,
                    "error_message": None
                },
                "system_env": {
                    "conda_env": "machine_learning",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "python_venv": sys.executable
                }
            }
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        # 로그파일 닫고, json 한 줄짜리 표준출력 재설정
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        _emit_json_line(result)
        return {
            "model_pt_path": str(model_path),
            "validation": val_score,
            "params": result[timestamp]["params"]
        }
    except Exception as e:
        import traceback
        errtype = type(e).__name__
        errmsg = str(e)
        trc = traceback.format_exc()
        end_time = time.time()
        result = {
            timestamp: {
                "validation_score": None,
                "train_score": None,
                "params": {
                    "test_size": args.test_size if "args" in locals() else None,
                    "num_epochs": args.epochs if "args" in locals() else None,
                    "batch_size": args.batch_size if "args" in locals() else None,
                    "lr": args.lr if "args" in locals() else None,
                    "save_path": str(run_dir) if "run_dir" in locals() else None,
                    "healthcheack": args.healthcheck if "args" in locals() else None
                },
                "runtime_info": {
                    "start_time": start_time_str if 'start_time_str' in locals() else None,
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                    "elapsed_time_sec": round(end_time - t0, 4)
                },
                "execution_status": {
                    "success": False,
                    "error_type": errtype,
                    "error_message": errmsg
                },
                "system_env": {
                    "conda_env": "machine_learning",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "python_venv": sys.executable
                }
            }
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = sys.__stdout__
        _emit_json_line(result)
        return {
            "model_pt_path": None,
            "validation": None,
            "params": result[timestamp]["params"]
        }

if __name__ == "__main__":
    autofinetuner_result()
