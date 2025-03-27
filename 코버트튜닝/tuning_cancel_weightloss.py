import os, re, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gluonnlp as nlp
from tqdm import tqdm
from kobert import get_pytorch_kobert_model
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
    text = re.sub(r'\S+@\S+|\b[\uac00-\ud7a3]{2,}\s*\uae30\uc790|\[EPA=.*?\]|\([^)]*= [^)]*\)', "", text)
    return text.strip()

def read_csv_with_encoding(csv_file):
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(csv_file, encoding=result['encoding'])

# -----------------------
# KoBERT 로드 및 토크나이저 초기화
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
bertmodel.to(device)

# -----------------------
# Dataset (뉴스 요약 여부 판단)
# -----------------------
class NewsSummaryDecisionDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, max_seq_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_length = max_seq_length
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
        tokens = self.tokenizer(body)[:self.max_seq_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = self.vocab[tokens] + [0] * (self.max_seq_length - len(tokens))
        return (
            torch.tensor(ids),
            torch.tensor(len(tokens)),
            torch.zeros(self.max_seq_length, dtype=torch.long),
            torch.tensor(label)
        )

# -----------------------
# 분류기 모델
# -----------------------
class NewsClassifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, ids, lengths, segs):
        mask = (ids != 0).long()
        _, pooled = self.bert(ids, mask, segs)
        return self.classifier(self.dropout(pooled))

# -----------------------
# 설정 및 데이터 로드
# -----------------------
csv_path = "/home/kororu/KoBERT/articles_summary3.csv"
checkpoint_root = os.path.join(os.path.dirname(csv_path), "checkpoints_doc_cancel", "epoch")
os.makedirs(checkpoint_root, exist_ok=True)

df = read_csv_with_encoding(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = NewsSummaryDecisionDataset(train_df, tokenizer, vocab, max_seq_length=128)
val_dataset = NewsSummaryDecisionDataset(val_df, tokenizer, vocab, max_seq_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = NewsClassifier(bertmodel).to(device)

train_labels = [label for _, label in train_dataset.samples]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts.astype(np.float32)
class_weights_tensor = torch.tensor(class_weights, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
korea_tz = pytz.timezone('Asia/Seoul')

ckpts = sorted([f for f in os.listdir(checkpoint_root) if f.endswith(".pt")])
start_epoch = int(ckpts[-1].split("_")[2]) if ckpts else 1
if ckpts:
    ckpt = torch.load(os.path.join(checkpoint_root, ckpts[-1]), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"\u25b6 Resuming from epoch {start_epoch}")

# -----------------------
# Early Stopping 설정
# -----------------------
best_f1_class0 = 0
patience = 3
patience_counter = 0

# -----------------------
# Training Loop with Validation
# -----------------------
num_epochs = 50

for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    total_loss = 0
    for ids, lengths, segs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        ids, lengths, segs, labels = ids.to(device), lengths.to(device), segs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(ids, lengths, segs)
        loss = criterion(outputs, labels)
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
    print(f"\u2714 Epoch {epoch} | Train Loss={avg_loss:.4f} | Saved checkpoint \u2192 {ckpt_path}")

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, lengths, segs, labels in tqdm(eval_loader, desc="Eval"):
            ids, lengths, segs, labels = ids.to(device), lengths.to(device), segs.to(device), labels.to(device)
            outputs = model(ids, lengths, segs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            pred = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().tolist()
            all_preds.extend(pred)
            all_labels.extend(labels.cpu().tolist())

    avg_val_loss = val_loss / len(eval_loader)
    print(f"\u2714 Epoch {epoch} | Val Loss={avg_val_loss:.4f}")
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f} | F1: {f1_score(all_labels, all_preds):.4f}")

    out_csv = os.path.join(checkpoint_root, f"epoch_{epoch:03d}_predictions_{ts}.csv")
    pd.DataFrame({
        "prediction": all_preds,
        "label": all_labels
    }).to_csv(out_csv, index=False)
    print(f"\u2714 Saved predictions \u2192 {out_csv}\n")

    # -----------------------
    # Early Stopping 체크
    # -----------------------
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    f1_class0 = report_dict['0']['f1-score']

    if f1_class0 > best_f1_class0:
        best_f1_class0 = f1_class0
        patience_counter = 0
        print(f"\u2714\ufe0f Improved F1 for class 0: {f1_class0:.4f} (Best so far)")
    else:
        patience_counter += 1
        print(f"\u23f3 No improvement in F1 for class 0 | Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered due to no improvement in class 0 F1-score.")
            break