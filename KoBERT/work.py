import os
import re
import chardet
import pandas as pd
import kss
import torch
import torch.nn.functional as F
from tqdm import tqdm
from kobert import get_pytorch_kobert_model
import gluonnlp as nlp

# 텍스트 클렌징 함수
def clean_text(text: str) -> str:
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b[가-힣]{2,}\s*기자\b', '', text)
    text = re.sub(r'\[EPA=.*?\]', '', text)
    text = re.sub(r'\([^)]*= [^)]*\)', '', text)
    return text.strip()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model, vocab = get_pytorch_kobert_model()
tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)

# 분류 모델 정의
class SentenceClassifier(torch.nn.Module):
    def __init__(self, bert, hidden_size=768, dropout_rate=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_size, 2)

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = (token_ids != 0).long()
        _, pooled = self.bert(token_ids, attention_mask, segment_ids)
        return self.classifier(self.dropout(pooled))

# 체크포인트 로드
checkpoint_dir = "/home/kororu/KoBERT/checkpoints2/epoch"
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
if not checkpoints:
    raise FileNotFoundError(f"❌ No .pt checkpoint files found in: {checkpoint_dir}")

latest_checkpoint = checkpoints[-1]
checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

model = SentenceClassifier(bert_model).to(device)

try:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load checkpoint: {checkpoint_path}\nError: {e}")

# 인코딩 감지 함수
def detect_encoding(filepath, n_bytes=10000):
    with open(filepath, "rb") as f:
        return chardet.detect(f.read(n_bytes))["encoding"]

# 토크나이징 함수
def tokenize(sentences, max_len=128):
    batch_ids, batch_seg, batch_len = [], [], []
    for s in sentences:
        tokens = tokenizer(clean_text(s))[: max_len-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = vocab[tokens]
        pad = [0] * (max_len - len(ids))
        batch_ids.append(ids + pad)
        batch_seg.append([0]*max_len)
        batch_len.append(len(ids))
    return (
        torch.tensor(batch_ids, dtype=torch.long).to(device),
        torch.tensor(batch_len, dtype=torch.long).to(device),
        torch.tensor(batch_seg, dtype=torch.long).to(device),
    )

# 문장 임베딩 추출 함수
def get_sentence_embeddings(sentences, max_seq_length=128):
    if not sentences:
        return torch.zeros((0, 768), device=device)
    ids, lengths, segs = tokenize(sentences, max_seq_length)
    model.eval()
    with torch.no_grad():
        _, pooled = bert_model(ids, (ids != 0).long(), segs)
    return pooled

# ✅ 처리할 모든 파일이 있는 디렉토리
input_dir = "/home/kororu/KoBERT/"

# ✅ 특정 패턴의 파일 리스트 가져오기
csv_files = [f for f in os.listdir(input_dir) if f.endswith("_processed.csv")]

for csv_file in csv_files:
    csv_path = os.path.join(input_dir, csv_file)

    # ✅ 데이터 로드
    df = pd.read_csv(csv_path, encoding=detect_encoding(csv_path))

    # ✅ 출력 파일명 생성 (요약 문장 수 표기 제거)
    base_filename = os.path.splitext(csv_file)[0]
    out_path = os.path.join(input_dir, f"{base_filename}_summary.csv")

    summaries = []

    # ✅ 요약 생성
    for body in tqdm(df["Body"].fillna(""), desc=f"Processing {csv_file}"):
        cleaned = clean_text(body)
        sents = kss.split_sentences(cleaned)
        if not sents:
            summaries.append("")
            continue

        body_emb = get_sentence_embeddings(sents)
        if body_emb.shape[0] == 0:
            summaries.append("")
            continue

        # ✅ row마다 요약 문장 수 동적으로 계산
        num_summary_sentences = min(8, max(3, len(body.split()) // 50))

        sum_emb = body_emb.mean(dim=0, keepdim=True)
        sims = torch.mm(F.normalize(body_emb, dim=1), F.normalize(sum_emb, dim=1).T)
        sims = sims.view(-1).tolist()

        selected_sentences = sorted(zip(sents, sims), key=lambda x: x[1], reverse=True)[:num_summary_sentences]
        selected_sentences.sort(key=lambda x: sents.index(x[0]))  # 원래 순서 복원
        selected_sentences = [s for s, _ in selected_sentences]
        summaries.append(" ".join(selected_sentences))

    # ✅ 결과 저장 (원본 row 보존)
    df["predicted_summary"] = summaries
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Summaries saved → {out_path}")
