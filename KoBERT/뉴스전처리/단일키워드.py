import pandas as pd
import cupy as cp
import re
from tqdm import tqdm
import os

# 저장 디렉터리 지정
output_dir = '/home/kororu/seyoung/KoBERT/뉴스전처리/'
os.makedirs(output_dir, exist_ok=True)

# 특수문자 제거
def normalize_text(text):
    return re.sub(r'[^가-힣a-zA-Z0-9]', ' ', text)

# 키워드 포함 여부 검사
def contains_keyword(text, keywords):
    normalized_text = normalize_text(text)
    for word in keywords:
        if normalize_text(word) in normalized_text:
            return True
    return False

# GPU 처리 함수
def process_with_gpu(df, target_words, must_keep_keywords):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        combined_text = ""
        if pd.notnull(row.get('body', None)):
            combined_text += row['body'] + " "
        if pd.notnull(row.get('title', None)):
            combined_text += row['title']

        if contains_keyword(combined_text, must_keep_keywords):
            results.append(cp.array([False]))  # keep
        elif contains_keyword(combined_text, target_words):
            results.append(cp.array([True]))   # discard
        else:
            results.append(cp.array([False]))  # keep
    return results

# 제외 키워드
target_words = [
    '중학생','고등학생', '부고', '화소', '오피스텔', '시세표', '테마파크', '스튜디오', '배우자',
    '자녀', '대단지', '분양', '아카데미', '출산', '결혼', '울음', '오락실', 'GSAT', '친환경', '선순환',
    '상생', '소상공인', '화질', '농산물', '축산물', '소비전력', '가짜뉴스','아주머니', '일자리', '자원봉사',
    '상위종목', '프라자', '추첨', '선착순', '싸구려', '소비효율', '실외기',
    '벽걸이','맥주','멤버십','주식부','학교','올림픽','여성','지스타','캠퍼스','추도식','업데이트','유네스코','체험','푸드'
]

# 반드시 keep할 키워드
must_keep_keywords = [
    '영업이익', '어닝', '순이익', '매출', '감익', '적자', '어닝 쇼크','협력','점유',
    '감산', '수출', '공급 부족', '공급 과잉', '환율', '변동성', '경영권', '합병', '파업', '증자','협업','동맹','협약'
]

# 전체 통합 결과를 담을 리스트
all_keep = []
all_discard = []

# 반복 처리할 연도 범위
for year in range(2019, 2026):
    file_path = f'/home/kororu/seyoung/KoBERT/005930_{year}_processed.csv'
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않음: {file_path}")
        continue

    print(f"\n[!] {year}년 파일 처리 중: {file_path}")
    df = pd.read_csv(file_path)

    # 컬럼 이름 소문자로 통일
    df.columns = [col.lower() for col in df.columns]

    results = process_with_gpu(df, target_words, must_keep_keywords)
    bool_mask = cp.array(results).astype(bool).get().flatten()
    discard_df = df[bool_mask]
    keep_df = df[~bool_mask]

    # 전체 통합 리스트에 추가
    all_discard.append(discard_df)
    all_keep.append(keep_df)

    # 연도별 keep 저장
    keep_df['year'] = keep_df['date'].astype(str).str[:4]
    for y, group in keep_df.groupby('year'):
        group.drop(columns='year').to_csv(os.path.join(output_dir, f'keep_{y}.csv'), index=False)

    print(f"→ {year}년 처리 완료")

# 전체 통합 CSV 저장
if all_keep:
    pd.concat(all_keep).to_csv(os.path.join(output_dir, 'keep_전체.csv'), index=False)
if all_discard:
    pd.concat(all_discard).to_csv(os.path.join(output_dir, 'discard_전체.csv'), index=False)

print("\n✅ 모든 연도 처리 및 통합 CSV 저장 완료!")
