import os, re, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import pytz
import chardet
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

# -----------------------
# 전처리 함수
# -----------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\S+@\S+|\b[가-힣]{2,}\s*기자|\[EPA=.*?\]|\([^)]*= [^)]*\)', "", text)
    return text.strip()

def read_csv_with_encoding(csv_file):
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(csv_file, encoding=result['encoding'])

# -----------------------
# KLUE-RoBERTa 로드
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# -----------------------
# Dataset 클래스
# -----------------------
class NewsSummaryDecisionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Dataset Build"):
            body = row.get('body', '')
            summary = row.get('summary', '')
            if not isinstance(body, str):
                body = ""
            if not isinstance(summary, str):
                summary = ""
            label = 1 if summary.strip() else 0
            if not body.strip():
                continue
            self.samples.append((body, label))
        print("총 샘플 수:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body, label = self.samples[idx]
        body = clean_text(body)
        encoding = self.tokenizer(
            body,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label)

# -----------------------
# 설정 및 데이터 로드
# -----------------------
csv_path = "/home/kororu/KoBERT/articles_summary3.csv"
checkpoint_root = os.path.join(os.path.dirname(csv_path), "checkpoints_doc_klue", "epoch")
os.makedirs(checkpoint_root, exist_ok=True)
best_model_path = os.path.join(checkpoint_root, "best_model.pt")

df = read_csv_with_encoding(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = NewsSummaryDecisionDataset(train_df, tokenizer)
val_dataset = NewsSummaryDecisionDataset(val_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -----------------------
# Weighted Loss
# -----------------------
train_labels = [label for _, label in train_dataset.samples]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts.astype(np.float32)
class_weights_tensor = torch.tensor(class_weights, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
korea_tz = pytz.timezone('Asia/Seoul')

# -----------------------
# Resume 학습 지원
# -----------------------
ckpts = sorted([f for f in os.listdir(checkpoint_root) if f.endswith(".pt") and "best_model" not in f])
start_epoch = int(ckpts[-1].split("_")[2]) if ckpts else 1
if ckpts:
    ckpt = torch.load(os.path.join(checkpoint_root, ckpts[-1]), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"Resuming from epoch {start_epoch}")

# -----------------------
# Training Loop with Early Stopping
# -----------------------
num_epochs = 50
patience = 3
best_f1_class1 = 0
epochs_no_improve = 0

for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    ts = datetime.now(korea_tz).strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(checkpoint_root, f"checkpoint_epoch_{epoch:03d}_{ts}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)
    print(f"✔ Epoch {epoch} | Train Loss={avg_loss:.4f} | Saved checkpoint → {ckpt_path}")

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(eval_loader, desc="Eval"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            pred = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).cpu().tolist()
            all_preds.extend(pred)
            all_labels.extend(labels.cpu().tolist())

    avg_val_loss = val_loss / len(eval_loader)
    print(f"✔ Epoch {epoch} | Val Loss={avg_val_loss:.4f}")

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f} | Weighted F1: {f1_score(all_labels, all_preds, average='weighted'):.4f} | Class 1 Recall: {report['1']['recall']:.4f}")

    # Early stopping based on class 1 f1-score
    current_f1_class1 = report['1']['f1-score']
    if current_f1_class1 > best_f1_class1:
        best_f1_class1 = current_f1_class1
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Improvement in class 1 F1-score. Saved best model → {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"⏳ No improvement in class 1 F1-score | Patience: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print("Early stopping triggered due to no improvement in class 1 F1-score.")
            break

    out_csv = os.path.join(checkpoint_root, f"epoch_{epoch:03d}_predictions_{ts}.csv")
    pd.DataFrame({
        "prediction": all_preds,
        "label": all_labels
    }).to_csv(out_csv, index=False)
    print(f"Saved predictions → {out_csv}\n")