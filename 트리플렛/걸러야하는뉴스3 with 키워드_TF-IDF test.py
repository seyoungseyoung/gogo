import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# ---------------------------
# 설정
# ---------------------------
article_folder_path = "/home/kororu/KoBERT"
top_percentiles_to_test = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ---------------------------
# 파일 불러오기
# ---------------------------
collected_files = [f for f in os.listdir(article_folder_path)
                   if f.startswith("collected_with_keyword_") and f.endswith(".csv")]

all_articles = []

for file in tqdm(collected_files, desc="파일 읽는 중"):
    path = os.path.join(article_folder_path, file)
    df = pd.read_csv(path)
    if 'Body' in df.columns and not df.empty:
        df = df[['Body', 'matched_keyword_triplets']].copy()
        df['source_file'] = file
        all_articles.append(df)
    else:
        print(f"⚠️ '{file}' 파일에 'Body' 컬럼이 없습니다.")

if not all_articles:
    raise ValueError("🚨 수집된 기사가 없습니다.")

all_articles_df = pd.concat(all_articles, ignore_index=True)

# ---------------------------
# TF-IDF 계산
# ---------------------------
vectorizer = TfidfVectorizer(max_features=1000, stop_words=[
    '것', '수', '등', '이후', '관련', '위해', '대해', '통해', '대한',
    '기자', '보도', '있다', '이번', '했다', '했다며'
])
tfidf_matrix = vectorizer.fit_transform(all_articles_df['Body'].fillna(''))
scores = tfidf_matrix.sum(axis=1).A1
all_articles_df['tfidf_score'] = scores

# ---------------------------
# 기준별 커트라인 근처 기사 출력
# ---------------------------
print("\n\n===== TF-IDF 기준별 커트라인 뉴스 예시 =====\n")

for p in top_percentiles_to_test:
    threshold = pd.Series(scores).quantile(p)
    
    # 커트라인 근처 기사만 선택 (±0.05 margin)
    margin = 0.05
    lower_bound = threshold - margin
    upper_bound = threshold + margin
    near_threshold_df = all_articles_df[
        (all_articles_df['tfidf_score'] >= lower_bound) &
        (all_articles_df['tfidf_score'] <= upper_bound)
    ].sort_values(by='tfidf_score')

    print(f"[Top {int(p * 100)}% 기준]")
    print(f"- 커트라인: {threshold:.4f}")
    print(f"- 커트라인 근처 기사 수: {len(near_threshold_df)}")
    print("- 대표 기사 2개 (50자 미리보기):")

    for i, row in enumerate(near_threshold_df.head(2).itertuples(), 1):
        body = row.Body[:50].strip().replace('\n', ' ')
        print(f"  ({i}) {body}...")

    print("-" * 50)

    # 저장 필요 시 주석 해제
    # output_file = os.path.join(article_folder_path, f"tfidf_cutline_{int(p * 100)}.csv")
    # near_threshold_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("✅ 커트라인 비교 완료.")
