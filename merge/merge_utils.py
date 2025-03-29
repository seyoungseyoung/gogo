import os
import pandas as pd
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import chardet

# 파일 인코딩 자동 감지
def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result['encoding']

# 임베딩 로딩
def load_embeddings(directory, file_end):
    print(f"Loading embeddings from: {directory}")
    embeddings = {}

    for file in os.listdir(directory):
        if file.endswith(file_end):
            file_path = os.path.join(directory, file)
            print(f"Reading embeddings from: {file_path}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error reading {file_path}: {e}")
                continue

            print(f"✅ Loaded JSON with {len(data)} entries")

            for item in data:
                key_id = item.get("ID")
                key_embedding = item.get("embedding") or item.get("Embedding")
                if key_id is None or key_embedding is None:
                    continue

                cleaned_id = str(key_id).strip().lower()
                embeddings[cleaned_id] = key_embedding

    print(f"Total embeddings loaded: {len(embeddings)}")
    print("Sample embedding keys:", list(embeddings.keys())[:5])
    return embeddings

# ID 정규화 함수
def normalize_id(raw_id):
    raw = str(raw_id).strip().lower().replace("_", "")
    if raw.isdigit():
        return raw.zfill(11)  # 예: 20190101001
    return raw

# 뉴스 CSV 불러오기
def load_processed_data(directory, embeddings):
    print(f"Loading processed data from: {directory}")
    all_files = [f for f in os.listdir(directory) if f.endswith(" (1).csv")]
    df_list = []

    for file in all_files:
        file_path = os.path.join(directory, file)
        print(f"Reading file: {file_path}")

        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        temp_df = pd.read_csv(file_path, encoding=encoding, dtype={"ID": str})

        temp_df = temp_df[temp_df["ID"].notnull()]
        temp_df["ID"] = temp_df["ID"].apply(normalize_id)
        temp_df["Date"] = pd.to_datetime(temp_df["Date"], format="%Y%m%d")

        print(f"Before filtering by ID, rows: {len(temp_df)}")
        temp_df = temp_df[temp_df["ID"].isin(embeddings)]
        print(f"After filtering by ID, rows: {len(temp_df)}")

        matched_ids = set(temp_df["ID"]) & set(embeddings)
        print(f"Matched ID count in {file}: {len(matched_ids)}")
        print("Sample processed IDs after normalization:", temp_df["ID"].unique()[:5])

        temp_df = temp_df[["Date", "Num_comment", "ID"]]
        df_list.append(temp_df)

    processed_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    print(f"Total rows loaded: {len(processed_df)}")
    return processed_df

# 날짜별 Top 5 ID 선택
def select_top_ids(processed_df, min_date):
    print("Selecting top 5 IDs per date...")
    selected_ids = {}
    fill_records = {}

    unique_dates = sorted(processed_df["Date"].unique())

    for date in unique_dates:
        date_str = date.strftime("%Y%m%d")
        top_ids = []
        fill_details = []

        temp_date = date
        while len(top_ids) < 5 and temp_date >= min_date:
            date_subset = processed_df[processed_df["Date"] == temp_date]
            sorted_subset = date_subset.sort_values("Num_comment", ascending=False)

            count_added = 0
            for _, row in sorted_subset.iterrows():
                id_str = str(row["ID"])
                if id_str not in top_ids:
                    top_ids.append(id_str)
                    count_added += 1
                if len(top_ids) >= 5:
                    break

            if count_added > 0:
                fill_details.append({
                    "date": temp_date.strftime("%Y%m%d"),
                    "count": count_added
                })

            temp_date -= timedelta(days=1)

        print(f"{date_str}: Selected {len(top_ids)} IDs")
        selected_ids[date_str] = top_ids[:5]
        fill_records[date_str] = {"sources": fill_details, "ids": top_ids[:5]}

    return selected_ids, fill_records

# ID 기반으로 임베딩 추출
def match_embeddings(selected_ids, embeddings, embedding_dim):
    print("Matching embeddings with selected IDs...")
    news_data = {}

    for date, ids in selected_ids.items():
        id_embeddings = [
            embeddings[i] if i in embeddings else [0.0] * embedding_dim
            for i in ids
        ]
        flat_embeddings = [val for sublist in id_embeddings for val in sublist]
        news_data[date] = flat_embeddings

    return news_data

# 재무 데이터 병합
def merge_with_financial(financial_path, news_data, output_path, embedding_dim):
    print(f"Merging news data with financial data from: {financial_path}")

    encoding = detect_encoding(financial_path)
    print(f"Detected encoding: {encoding}")
    financial_df = pd.read_csv(financial_path, encoding=encoding)
    financial_df["Date"] = pd.to_datetime(financial_df["Date"], format="%Y-%m-%d")

    news_cols = [f"News_{i:02d}_{j:03d}" for i in range(1, 6) for j in range(1, embedding_dim + 1)]
    news_df = pd.DataFrame(columns=["Date"] + news_cols)

    for date_str, embedding in tqdm(news_data.items(), desc="Merging news data", unit="date"):
        shifted_date = (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        row_data = [shifted_date] + embedding
        news_df.loc[len(news_df)] = row_data

    news_df["Date"] = pd.to_datetime(news_df["Date"], format="%Y-%m-%d")
    merged_df = financial_df.merge(news_df, on="Date", how="left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged dataset saved to: {output_path}")
