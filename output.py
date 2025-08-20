import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from torch.optim import AdamW
from dataset import BertDataset, from_csv

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

# Enable gradient checkpointing to reduce memory usage
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

model.to(device)

# Load data
train_texts, train_labels = from_csv("/home/jeongyuseong/바탕화면/오픈소스경진대회/AutoFineTuner/datas/train.csv")

# Use smaller batch size to avoid OOM
TRAIN_BATCH_SIZE = 16
MAX_LEN = 512

train_dataset = BertDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
train_dataset.set_loaders(batch_size=TRAIN_BATCH_SIZE)
train_dataloader, valid_dataloader = train_dataset.get_loaders()

# Optimizer and scheduler
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.1},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
epochs = 3

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=5e-5,
                  eps=1e-8)
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training
train_loss_set = []

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

torch.cuda.empty_cache()

for _ in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

        train_loss_set.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tr_loss += loss.item()
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / max(1, nb_tr_steps)))

    # Validation
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0

    with torch.no_grad():
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy / max(1, nb_eval_steps)))

# Inference on test set
test = pd.read_csv("./datas/test.csv")
test_texts = test["paragraph_text"].to_list()
labels = [0] * len(test_texts)
ids = test["ID"].tolist()

TEST_BATCH_SIZE = 32  # smaller batch size to prevent OOM during inference
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
dataset = BertDataset(test_texts, labels, apply_preprocess=False, tokenizer=tokenizer, max_len=MAX_LEN)
dataset.set_loaders(batch_size=TEST_BATCH_SIZE, split_ratio=1.0)
test_loader, _ = dataset.get_loaders()

for batch in test_loader:
    input_ids, attn_mask, labels = batch
    print(input_ids.shape, attn_mask.shape, labels.shape)

def softmax(logits):
    e = np.exp(logits)
    return e / np.sum(e)

all_probs = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attn_mask, labels = [t.to(device) for t in batch]
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(input_ids, attention_mask=attn_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

assert len(ids) == len(all_probs), f"Mismatch: {len(ids)} IDs vs {len(all_probs)} predictions"

submission = pd.DataFrame({
    "ID": ids,
    "generated": all_probs
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 저장 완료")

plt.figure(figsize=(15, 8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()