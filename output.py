import argparse
import sys, json, time, pickle, os
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets


def _save_model_generic(model, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(getattr(model, "state_dict", lambda: model)(), path)
    except Exception:
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--train_path", type=str, default="./data/CIFAR10/")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--healthcheck", action="store_true")
    args, _ = parser.parse_known_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.save_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"{timestamp}.log"
    sys.stdout = open(log_path, "w", encoding="utf-8")

    # STEP 1: LOADING DATASET
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = dsets.CIFAR10(root=args.train_path, train=True, transform=transform_train, download=True)
    test_dataset = dsets.CIFAR10(root=args.train_path, train=False, transform=transform_test)

    reduced_train_dataset = [(images, labels) for images, labels in train_dataset if labels < 3]
    reduced_test_dataset = [(images, labels) for images, labels in test_dataset if labels < 3]

    print("The number of training images : ", len(reduced_train_dataset))
    print("The number of test images : ", len(reduced_test_dataset))

    # STEP 2: MAKING DATASET ITERABLE
    train_loader = torch.utils.data.DataLoader(dataset=reduced_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=reduced_test_dataset, batch_size=100, shuffle=False)

    # STEP 3: CREATE MODEL CLASS (VGG16)
    cfg = [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']

    class VGG(nn.Module):
        def __init__(self, num_layer=1):
            super(VGG, self).__init__()
            self.VGG16 = self._make_layers(cfg)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = self._make_classifier(num_layer)

        def forward(self, x):
            out = self.VGG16(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

        def _make_classifier(self, num_layer: int):
            layer = []
            for _ in range(num_layer):
                layer += [nn.Linear(512, 512), nn.ReLU(inplace=True)]
            layer += [nn.Linear(512, 3)]
            return nn.Sequential(*layer)

        def _make_layers(self, cfg):
            layers = []
            in_channels = 3
            for x in cfg:
                if x == 'MP':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                    in_channels = x
            return nn.Sequential(*layers)

    # STEP 4: INSTANTIATE MODEL CLASS
    model = VGG()
    num_total_params = sum(p.numel() for p in model.parameters())
    print("The number of parameters : ", num_total_params)

    # STEP 5: INSTANTIATE LOSS CLASS
    criterion = nn.CrossEntropyLoss()

    # STEP 6: INSTANTIATE OPTIMIZER CLASS
    learning_rate = 1e-2
    momentum = 0.9
    weight_decay = 5e-4

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # STEP 7: TRAIN THE MODEL
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        elapsed_time = time.time() - start

        print(f"Epochs: {epoch}. Loss: {epoch_loss:.4f}. Accuracy: {epoch_acc:.2f}. Elapsed time: {elapsed_time:.2f} sec")

    # Save model
    model_path = run_dir / "model.pt"
    _save_model_generic(model, model_path)

    # Prepare and save result.json
    result = {
        "model_pt_path": str(model_path),
        "validation_score": epoch_acc,
        "params": {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size
        }
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    _emit_json_line(result)
    return result

autofinetuner_result()