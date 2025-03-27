import pandas as pd
from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm
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
# 키워드 조합 추출 함수
# -----------------------
def extract_keyword_pairs(texts, tokenizer, min_len=2):
    pair_list = []
    for text in tqdm(texts, desc="키워드 쌍 추출 중"):
        if not isinstance(text, str):
            continue
        nouns = tokenizer.nouns(text)
        filtered = [word for word in nouns if len(word) >= min_len]
        pairs = combinations(sorted(set(filtered)), 2)
        pair_list.extend(pairs)
    return pair_list

# -----------------------
# 주요 실행 로직
# -----------------------
def analyze_keyword_pairs(csv_path):
    df = read_csv_with_encoding(csv_path)

    okt = Okt()

    # 요약 있음 / 없음 나누기
    with_summary = df[df['summary'].notnull() & (df['summary'].str.strip() != '')]['body'].dropna()
    without_summary = df[df['summary'].isnull() | (df['summary'].str.strip() == '')]['body'].dropna()

    print(f"요약 있음 기사 수: {len(with_summary)}")
    print(f"요약 없음 기사 수: {len(without_summary)}")

    # 키워드 쌍 추출
    with_pairs = extract_keyword_pairs(with_summary, okt)
    without_pairs = extract_keyword_pairs(without_summary, okt)

    with_counter = Counter(with_pairs)
    without_counter = Counter(without_pairs)

    # 모든 키워드 쌍에 대해 비교
    full_pair_stats = []
    for pair, without_count in without_counter.items():
        with_count = with_counter.get(pair, 0)
        if with_count == 0 and without_count >= 3:
            full_pair_stats.append({
                "keyword1": pair[0],
                "keyword2": pair[1],
                "without_summary_count": without_count,
                "with_summary_count": with_count,
                "frequency_gap": without_count - with_count
            })

    # 정렬
    full_pair_stats = sorted(full_pair_stats, key=lambda x: x["without_summary_count"], reverse=True)

    print("\n🧩 요약 없음 뉴스에만 등장하는 키워드 조합 Top 30:")
    for item in full_pair_stats[:30]:
        print(f"{item['keyword1']}, {item['keyword2']}: {item['without_summary_count']}")

    # 저장
    df_result = pd.DataFrame(full_pair_stats)
    df_result.to_csv("keyword_pair_stats.csv", index=False)
    print("\n📁 결과 저장됨 → keyword_pair_stats.csv")

# 예시 실행
csv_path = "/home/kororu/KoBERT/articles_summary24.csv"  # ← 네 경로에 맞게 수정
analyze_keyword_pairs(csv_path)
