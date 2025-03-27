import os
import re
import chardet
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------- #
# 텍스트 정제 함수
# ----------------------- #
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\S+@\S+|\b[가-힣]{2,}\s*기자|\[EPA=.*?\]|\([^)]*= [^)]*\)', "", text)
    return text.strip()

# ----------------------- #
# 설정
# ----------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/kororu/KoBERT/checkpoints_doc_klue/epoch/best_model.pt"
model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"✅ Loaded model from {model_path}")

# ----------------------- #
# 예측 함수
# ----------------------- #
def predict(text, model, tokenizer, max_len=128):
    cleaned = clean_text(text)
    if not cleaned:
        return 0, 0.0
    encoding = tokenizer(
        cleaned,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(output.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()
    return pred, confidence

# ----------------------- #
# 테스트 데이터셋 로드
# ----------------------- #
def detect_encoding(filepath, n_bytes=10000):
    with open(filepath, "rb") as f:
        return chardet.detect(f.read(n_bytes))["encoding"]

csv_path = "/home/kororu/KoBERT/test_byte.csv"
df = pd.read_csv(csv_path, encoding=detect_encoding(csv_path))

# ----------------------- #
# 예측 실행
# ----------------------- #
predicted_labels = []
confidences = []

for body in tqdm(df["body"].fillna(""), desc="Predicting"):
    label, conf = predict(body, model, tokenizer)
    predicted_labels.append(label)
    confidences.append(conf)

df["predicted_label"] = predicted_labels
df["confidence"] = confidences

out_path = "/home/kororu/KoBERT/test_byte_predictions.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ Predictions saved → {out_path}")