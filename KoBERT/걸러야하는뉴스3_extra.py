import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# ---------------------------
# 설정
# ---------------------------
article_folder_path = "/home/kororu/KoBERT"
tfidf_top_percentile = 0.2  # 상위 20%는 '살려야함'으로 분류

# ---------------------------
# 키워드 정의
# ---------------------------
must_keep_keywords = [
    '영업이익', '어닝', '순이익', '매출', '감익', '적자', '어닝 쇼크',
    '급감', '급등', '하락세', '상승세', '감산', '수출', '출하량',
    '공급 부족', '공급 과잉', '환율', '투자계획', '변동성', '매도세', '매수세',
    '경영권', '합병', '파업', '증자'
]


irrelevant_keywords = [
    '학생', '사회공헌', '나눔', '봉사활동', '장학금',
    '사내 행사', '수상', '시상식', '축사', '기념식', '봉사', '사보', '행사 개최',
    '캠페인', '기념행사', '워크숍', '사보발간', '인터뷰', '기고문',
    '후원', '사회적기업', '멘토링', '교육기부', '문화행사',
    '자원봉사', '축제', '소셜벤처', '시민참여', '지역사회',
    '전시회', '사진전', '사내소식', '공로상', '사회적책임', '나눔활동',
    '공헌활동', '기업문화', '체험활동', '행사참여', '이벤트',
    '문화공연', '홍보대사', '사내동호회', '환경정화', '지역행사',
    '스폰서', '참여행사'
]

training_keywords = [
    '아카데미', '인재양성', '수료생', '양성과정', '교육생', '합숙',
    '개발자', '교육 프로그램', '커리큘럼', '취업 연계', 'SSA', 'SSAFY', 'AIVLE', '에이블 스쿨',
    '부트캠프', '교육센터', '온보딩 교육', '신입교육', '기술교육',
    '직무교육', '사내교육', '리더십 교육', '연수과정', '직원 연수',
    '직무역량 강화', '기초교육', '현장실습', '취업캠프', '역량개발',
    '멘토링 프로그램', '온라인 강의', 'MOOC', '학습과정', '집체교육',
    '교육과정 개설', '테크 캠프', '코딩 교육', 'AI 아카데미',
    '사내강사', '교육이수', '자격증 과정', 'HRD 프로그램'
]

# ---------------------------
# 함수 정의
# ---------------------------
def is_must_keep(text):
    if not isinstance(text, str):
        return False
    return any(keyword in text for keyword in must_keep_keywords)

def is_irrelevant_context(text):
    if not isinstance(text, str):
        return False
    return any(keyword in text for keyword in irrelevant_keywords)

def is_training_news(text):
    if not isinstance(text, str):
        return False
    return any(keyword in text for keyword in training_keywords)

def final_judgment(row):
    body = row.Body
    # must keep 키워드가 있다면 우선 '살려야함'으로 처리 (우선순위 변경)
    if row.recovery_candidate:
        return '살려야함'
    elif is_irrelevant_context(body) or is_training_news(body):
        return '그대로 걸러야함'
    else:
        return row.judgment


# ---------------------------
# 파일 불러오기
# ---------------------------
collected_files = [f for f in os.listdir(article_folder_path)
                   if f.startswith("005930_20") and f.endswith(".csv")]

all_articles = []

for file in tqdm(collected_files, desc="📂 파일 읽는 중"):
    path = os.path.join(article_folder_path, file)
    df = pd.read_csv(path)
    if 'Body' in df.columns and not df.empty:
        df = df[['Title','Date','Link','Press','Body','Emotion','Comment_body','Comment_date','Comment_recomm','Comment_unrecomm','Num_comment','ID','version']].copy()
        df['source_file'] = file
        all_articles.append(df)
    else:
        print(f"⚠️ '{file}' 파일에 'Body' 컬럼이 없습니다.")

if not all_articles:
    raise ValueError("🚨 수집된 기사가 없습니다. CSV 파일과 컬럼을 다시 확인해주세요.")
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
threshold = pd.Series(scores).quantile(tfidf_top_percentile)

all_articles_df['tfidf_score'] = scores
all_articles_df['judgment'] = all_articles_df['tfidf_score'].apply(
    lambda x: '살려야함' if x >= threshold else '그대로 걸러야함'
)

# ---------------------------
# 실적 키워드 복구 판단 (tqdm 적용)
# ---------------------------
print("🔍 실적/위기 키워드 확인 중...")
all_articles_df['recovery_candidate'] = [
    is_must_keep(text) for text in tqdm(all_articles_df['Body'], desc="💡 키워드 복구 판단")
]

# ---------------------------
# 최종 판단 (tqdm 적용)
# ---------------------------
print("🧠 최종 필터 판단 중...")
all_articles_df['judgment'] = [
    final_judgment(row) for row in tqdm(all_articles_df.itertuples(), total=len(all_articles_df), desc="📝 최종 판단 적용")
]

# ---------------------------
# 결과 저장
# ---------------------------
output_file_keep = os.path.join(article_folder_path, "classified_articles_keep_extra.csv")
output_file_discard = os.path.join(article_folder_path, "classified_articles_discard_extra.csv")

all_articles_df[all_articles_df['judgment'] == '살려야함'].to_csv(output_file_keep, index=False, encoding='utf-8-sig')
all_articles_df[all_articles_df['judgment'] == '그대로 걸러야함'].to_csv(output_file_discard, index=False, encoding='utf-8-sig')

print(f"\n✅ TF-IDF + 키워드 필터 적용 완료:")
print(f" - 살려야함: {output_file_keep}")
print(f" - 걸러야함: {output_file_discard}")