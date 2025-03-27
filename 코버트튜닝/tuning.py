import os, re, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gluonnlp as nlp
from tqdm import tqdm
import kss
from kobert import get_pytorch_kobert_model
from datetime import datetime
import pytz
from sklearn.metrics import classification_report, accuracy_score, f1_score
import chardet
import hashlib

# -----------------------
# 전처리 함수
# -----------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    text = re.sub(r'\S+@\S+|\b[가-힣]{2,}\s*기자|\[EPA=.*?\]|\([^)]*= [^)]*\)', "", text)
    return text.strip()

def split_into_sentences(text):
    cleaned = clean_text(text)
    return kss.split_sentences(cleaned) if cleaned else []

def read_csv_with_encoding(csv_file):
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(csv_file, encoding=result['encoding'])

# -----------------------
# KoBERT 로드
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
bertmodel.to(device)

# -----------------------
# 토크나이징 및 임베딩
# -----------------------
def tokenize_batch(sentences, tokenizer, vocab, max_seq_length=128):
    batch_ids, batch_len, batch_seg = [], [], []
    for sent in sentences:
        tokens = tokenizer(clean_text(sent))[: max_seq_length-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = vocab[tokens] + [0]*(max_seq_length - len(tokens))
        batch_ids.append(ids)
        batch_len.append(len(tokens))
        batch_seg.append([0]*max_seq_length)
    return (
        torch.tensor(batch_ids).to(device),
        torch.tensor(batch_len).to(device),
        torch.tensor(batch_seg).to(device),
    )

def get_sentence_embeddings(sentences, tokenizer, vocab, max_seq_length=128):
    if not sentences:
        return torch.zeros((0, 768), device=device)
    ids, lengths, segs = tokenize_batch(sentences, tokenizer, vocab, max_seq_length)
    bertmodel.eval()
    with torch.no_grad():
        _, pooled = bertmodel(ids, (ids != 0).long(), segs)
    return pooled

# -----------------------
# 임베딩 캐시 기능
# -----------------------
CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_hash_for_sentences(sentences):
    # 문장들을 구분자("||")로 연결한 후 MD5 해시 생성
    concatenated = "||".join(sentences)
    return hashlib.md5(concatenated.encode("utf-8")).hexdigest()

def get_cached_sentence_embeddings(sentences, tokenizer, vocab, max_seq_length=128):
    key = get_hash_for_sentences(sentences)
    cache_file = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(cache_file):
        return torch.load(cache_file, map_location=device)
    else:
        emb = get_sentence_embeddings(sentences, tokenizer, vocab, max_seq_length)
        torch.save(emb, cache_file)
        return emb

# -----------------------
# Dataset (Top‑K 기반 레이블링)
# -----------------------
class ExtractiveSummarizationDataset(Dataset):
    def __init__(self, df, tokenizer, vocab, max_seq_length=128, top_k=3):
        self.samples = []
        # 문서 단위로 반복
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Dataset Build"):
            body = split_into_sentences(row.get('body',''))
            summary = split_into_sentences(row.get('summary',''))
            if not body:
                continue
            if not summary:
                # summary가 공란이면 해당 문서의 모든 문장에 label 0
                self.samples += [(s, 0) for s in body]
            else:
                # summary가 있으면, 캐시된 임베딩을 사용하여 Top‑K 문장에 label 1 할당
                sum_emb = get_cached_sentence_embeddings(summary, tokenizer, vocab, max_seq_length)
                body_emb = get_cached_sentence_embeddings(body, tokenizer, vocab, max_seq_length)
                sims = torch.mm(F.normalize(body_emb, p=2, dim=1),
                                  F.normalize(sum_emb, p=2, dim=1).T).max(dim=1)[0]
                topk = torch.topk(sims, min(top_k, len(sims))).indices.tolist()
                for idx, sent in enumerate(body):
                    label = 1 if idx in topk else 0
                    self.samples.append((sent, label))
        print("총 샘플 수:", len(self.samples))

    def __len__(self): 
        return len(self.samples)
    def __getitem__(self, idx):
        sent, label = self.samples[idx]
        tokens = tokenizer(clean_text(sent))[:126]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = vocab[tokens] + [0]*(128 - len(tokens))
        return (
            torch.tensor(ids),
            torch.tensor(len(tokens)),
            torch.zeros(128, dtype=torch.long),
            torch.tensor(label),
            sent
        )

# -----------------------
# 분류기 모델
# -----------------------
class SentenceClassifier(nn.Module):
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
# 설정
# -----------------------
csv_path = "/home/kororu/KoBERT/articles_summary3.csv"
checkpoint_root = os.path.join(os.path.dirname(csv_path), "checkpoints_topk", "epoch")
os.makedirs(checkpoint_root, exist_ok=True)

df = read_csv_with_encoding(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ExtractiveSummarizationDataset(train_df, tokenizer, vocab, top_k=3)
val_dataset = ExtractiveSummarizationDataset(val_df, tokenizer, vocab, top_k=3)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 모델 세팅
model = SentenceClassifier(bertmodel).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
korea_tz = pytz.timezone('Asia/Seoul')

# Resume 기능
ckpts = sorted([f for f in os.listdir(checkpoint_root) if f.endswith(".pt")])
start_epoch = int(ckpts[-1].split("_")[2]) if ckpts else 1
if ckpts:
    ckpt = torch.load(os.path.join(checkpoint_root, ckpts[-1]), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"▶ Resuming from epoch {start_epoch}")

# -----------------------
# Training Loop with Validation
# -----------------------
for epoch in range(start_epoch, 101):
    model.train()
    total_loss = 0
    for ids, lengths, segs, labels, _ in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        ids, lengths, segs, labels = ids.to(device), lengths.to(device), segs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, lengths, segs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    ts = datetime.now(korea_tz).strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(checkpoint_root, f"checkpoint_epoch_{epoch:03d}_{ts}.pt")
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
    print(f"✔ Epoch {epoch} | Train Loss={avg_loss:.4f} | Saved checkpoint → {ckpt_path}")

    # -----------------------
    # Validation (문장 분류 평가)
    # -----------------------
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    preds = []

    with torch.no_grad():
        for ids, lengths, segs, labels, sents in tqdm(eval_loader, desc="Eval"):
            ids, lengths, segs, labels = ids.to(device), lengths.to(device), segs.to(device), labels.to(device)
            outputs = model(ids, lengths, segs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().tolist()
            prob_1 = probs[:, 1].cpu().tolist()
            gold = labels.cpu().tolist()

            all_preds.extend(pred)
            all_labels.extend(gold)
            preds.extend([
                {"sentence": s, "predict": p, "prob_1": prob, "gold": g}
                for s, p, prob, g in zip(sents, pred, prob_1, gold)
            ])

    avg_val_loss = val_loss / len(eval_loader)
    print(f"✔ Epoch {epoch} | Val Loss={avg_val_loss:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f} | F1: {f1_score(all_labels, all_preds):.4f}")

    out_csv = os.path.join(checkpoint_root, f"epoch_{epoch:03d}_predictions_{ts}.csv")
    pd.DataFrame(preds).to_csv(out_csv, index=False)
    print(f"✔ Saved predictions → {out_csv}\\n")

# -----------------------
# 문서 수준 요약 품질 평가 함수 (코사인 유사도 기반)
# -----------------------
def evaluate_summary_quality(df, model, tokenizer, vocab, max_seq_length=128, top_k=3):
    model.eval()
    cosine_sims = []
    for _, row in df.iterrows():
        body_text = row.get('body', '')
        summary_text = row.get('summary', '')
        if not summary_text.strip():
            continue  # 요약 가치 없는 문서는 건너뛰기
        body_sents = split_into_sentences(body_text)
        if not body_sents:
            continue
        summary_sents = split_into_sentences(summary_text)
        # 문서의 summary 임베딩 (평균)
        sum_emb = get_cached_sentence_embeddings(summary_sents, tokenizer, vocab, max_seq_length)
        if sum_emb.size(0) == 0:
            continue
        summary_avg = sum_emb.mean(dim=0, keepdim=True)  # (1, 768)
        # body 문장의 임베딩 (캐시 사용)
        body_emb = get_cached_sentence_embeddings(body_sents, tokenizer, vocab, max_seq_length)
        # 모델을 통해 각 문장의 중요도 확률 얻기
        ids, lengths, segs = tokenize_batch(body_sents, tokenizer, vocab, max_seq_length)
        with torch.no_grad():
            outputs = model(ids, lengths, segs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # label 1 확률
        # Top‑K 문장 선택
        topk_indices = torch.topk(probs, min(top_k, len(probs))).indices.tolist()
        extracted_emb = body_emb[topk_indices].mean(dim=0, keepdim=True)
        # 코사인 유사도 계산
        cos_sim = F.cosine_similarity(extracted_emb, summary_avg).item()
        cosine_sims.append(cos_sim)
    if cosine_sims:
        avg_cos_sim = sum(cosine_sims)/len(cosine_sims)
        print(f"Average cosine similarity between extracted summary and ground truth: {avg_cos_sim:.4f}")
    else:
        print("No valid documents for summary quality evaluation.")

# 문서 수준 요약 품질 평가 (예: 학습 후 검증 데이터에 대해)
evaluate_summary_quality(val_df, model, tokenizer, vocab)
