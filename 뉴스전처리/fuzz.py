import pandas as pd
from rapidfuzz import fuzz
import os

base_dir = r"C:\Users\tpdud\code\gogo\Database\뉴스키워드전처리"
save_dir = os.path.join(base_dir, "fuzz")
os.makedirs(save_dir, exist_ok=True)

discarded_rows = []

def deduplicate_titles(df, threshold=80):
    result = []
    for date, group in df.groupby('date'):
        titles = group['title'].tolist()
        kept = []
        discarded = []

        for idx, title in enumerate(titles):
            is_duplicate = False
            for kept_title in kept:
                if fuzz.ratio(title, kept_title) >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(title)
            else:
                discarded.append(title)

        kept_df = group[group['title'].isin(kept)]
        discarded_df = group[group['title'].isin(discarded)]
        result.append(kept_df)
        discarded_rows.append(discarded_df)

    return pd.concat(result, ignore_index=True)

for year in range(2019, 2026):
    file_path = os.path.join(base_dir, f"keep_{year}.csv")
    df = pd.read_csv(file_path)

    if 'date' not in df.columns or 'title' not in df.columns:
        print(f"{file_path}에 'date'나 'title' 칼럼이 없습니다.")
        continue

    cleaned_df = deduplicate_titles(df)
    save_path = os.path.join(save_dir, f"keep_{year}.csv")
    cleaned_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"{year}년 파일 저장 완료: {save_path}")

# 모든 연도에서 제거된 데이터 하나로 합치기
if discarded_rows:
    discarded_df = pd.concat(discarded_rows, ignore_index=True)
    discard_path = os.path.join(save_dir, "discard.csv")
    discarded_df.to_csv(discard_path, index=False, encoding='utf-8-sig')
    print(f"삭제된 항목 저장 완료: {discard_path}")
else:
    print("삭제된 항목이 없습니다.")
