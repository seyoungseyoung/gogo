import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm
import os
import chardet
from itertools import combinations

# -----------------------
# 파일 로딩 함수
# -----------------------
def read_csv_with_encoding(csv_file):
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(csv_file, encoding=result['encoding'])

# -----------------------
# 키워드쌍 포함 여부 확인 함수 (수집 기준과 동일하게)
# -----------------------
def article_contains_keyword_pair(text, valid_pairs_set, tokenizer, min_len=2):
    if not isinstance(text, str):
        return False
    nouns = tokenizer.nouns(text)
    filtered = [word for word in nouns if len(word) >= min_len]
    pairs = combinations(sorted(set(filtered)), 2)
    for pair in pairs:
        if pair in valid_pairs_set:
            return True
    return False

# -----------------------
# 실행 함수
# -----------------------
def collect_articles_with_keyword_pairs(pair_csv_path, article_folder_path, output_path):
    # keyword_pair_stats.csv 기준으로 수집
    pair_df = pd.read_csv(pair_csv_path)
    keyword_pairs = set(zip(pair_df['keyword1'], pair_df['keyword2']))  # set으로 바꿔서 빠르게 비교

    tokenizer = Okt()
    collected_articles = []

    for year in range(2019, 2026):
        file_path = os.path.join(article_folder_path, f"005930_{year}_processed.csv")
        print(f"\n📂 처리 중: {file_path}")
        df = read_csv_with_encoding(file_path)

        # Body 컬럼 찾기 (대소문자 구분 없이)
        body_column_name = next((col for col in df.columns if col.lower() == 'body'), None)
        if not body_column_name:
            print(f"⚠️ 'body' 관련 컬럼 없음. 실제 컬럼들: {df.columns}")
            continue

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{year} 기사 필터링"):
            body = row[body_column_name]
            if article_contains_keyword_pair(body, keyword_pairs, tokenizer):
                collected_articles.append(row)

    if collected_articles:
        result_df = pd.DataFrame(collected_articles)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 관련 뉴스 저장 완료: {output_path}")
    else:
        print("❌ 수집된 기사가 없습니다.")

# 실행 예시
pair_csv_path = "keyword_pair_stats.csv"  # ← analyze_keyword_pairs로 만든 쌍
article_folder_path = "/home/kororu/KoBERT"
output_path = "collected_articles_with_keywords.csv"

collect_articles_with_keyword_pairs(pair_csv_path, article_folder_path, output_path)
