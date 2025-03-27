import pandas as pd
import os
import chardet
import MeCab
from tqdm import tqdm
from itertools import combinations

# -----------------------
# 파일 로딩 함수
# -----------------------
def read_csv_with_encoding(csv_file):
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(csv_file, encoding=result['encoding'])

# -----------------------
# 명사 추출 함수 (MeCab 사용)
# -----------------------
tagger = MeCab.Tagger()

def extract_nouns(text, min_len=2):
    if not isinstance(text, str):
        return []
    parsed = tagger.parse(text)
    nouns = []
    for line in parsed.splitlines():
        if line == 'EOS':
            break
        try:
            word, features = line.split('\t')
            pos = features.split(',')[0]
            if pos in ['NNG', 'NNP'] and len(word) >= min_len:
                nouns.append(word)
        except ValueError:
            continue
    return sorted(set(nouns))

# -----------------------
# 키워드 3개 조합 포함 여부 확인 함수
# -----------------------
def article_contains_keyword_triplet(text, valid_triplets_set, min_len=2):
    nouns = extract_nouns(text, min_len)
    # valid_triplets_set에 들어있는 키워드 조합은 모두 정렬된 튜플임
    for triplet in combinations(nouns, 3):
        if triplet in valid_triplets_set:
            return True
    return False

# -----------------------
# 실행 함수
# -----------------------
def collect_articles_with_keyword_triplets(triplet_csv_path, article_folder_path, output_path):
    # 첫 번째 코드의 출력 파일 (keyword_triplet_stats.csv)을 기준으로 수집
    triplet_df = pd.read_csv(triplet_csv_path)
    keyword_triplets = set(zip(triplet_df['keyword1'], triplet_df['keyword2'], triplet_df['keyword3']))
    
    collected_articles = []

    # 예시로 2019년부터 2025년까지의 기사를 처리
    for year in range(2019, 2026):
        file_path = os.path.join(article_folder_path, f"005930_{year}_processed.csv")
        print(f"\n📂 처리 중: {file_path}")
        if not os.path.exists(file_path):
            print(f"⚠️ 파일이 존재하지 않습니다: {file_path}")
            continue
        df = read_csv_with_encoding(file_path)

        # 'body' 컬럼 찾기 (대소문자 구분 없이)
        body_column_name = next((col for col in df.columns if col.lower() == 'body'), None)
        if not body_column_name:
            print(f"⚠️ 'body' 컬럼을 찾을 수 없습니다. 실제 컬럼들: {df.columns}")
            continue

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{year} 기사 필터링"):
            body = row[body_column_name]
            if article_contains_keyword_triplet(body, keyword_triplets):
                collected_articles.append(row)

    if collected_articles:
        result_df = pd.DataFrame(collected_articles)
        result_df.to_csv(output_path, index=False)
        print(f"\n✅ 관련 뉴스 저장 완료: {output_path}")
    else:
        print("❌ 수집된 기사가 없습니다.")

# 실행 예시
triplet_csv_path = "3keywords.csv"  # 첫 번째 코드의 출력 파일
article_folder_path = "/home/kororu/KoBERT"
output_path = "collected_articles_with_keyword_triplets.csv"

collect_articles_with_keyword_triplets(triplet_csv_path, article_folder_path, output_path)
