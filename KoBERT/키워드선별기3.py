import pandas as pd
from collections import Counter
from tqdm import tqdm
import chardet
import MeCab
import os
import pickle
import gzip
import psutil
import cupy as cp
import multiprocessing
import itertools

# -----------------------
# 전역 설정: 체크포인트 파일 저장 디렉터리
# -----------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # checkpoints 폴더가 없으면 생성

# -----------------------
# 파일 로딩 함수
# -----------------------
def read_csv_with_encoding(csv_file):
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(csv_file, encoding=result['encoding'])

# -----------------------
# 명사 추출 함수 (전역 태거 재사용)
# -----------------------
# 메인 프로세스용 태거
tagger = MeCab.Tagger()

def extract_nouns(text, min_len=2):
    parsed = tagger.parse(text)
    nouns = set()
    for line in parsed.splitlines():
        if line == 'EOS':
            break
        try:
            word, features = line.split('\t')
            pos = features.split(',')[0]
            if pos in ['NNG', 'NNP'] and len(word) >= min_len:
                nouns.add(word)
        except ValueError:
            continue
    return sorted(nouns)

# -----------------------
# 명사 사전 캐시 활용
# -----------------------
def build_or_load_noun_dictionary(texts):
    cache_path = "noun_mapping.pkl"
    if os.path.exists(cache_path):
        print(f"\n📦 명사 사전 캐시 발견 → {cache_path} 로드 중...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data['noun2id'], data['id2noun']

    print("\n🆕 명사 사전 캐시 없음 → 새로 생성 중...")
    noun_set = set()
    for text in tqdm(texts, desc="🔍 명사 사전 생성 중"):
        noun_set.update(extract_nouns(text))
    noun2id = {noun: i for i, noun in enumerate(sorted(noun_set))}
    id2noun = {i: noun for noun, i in noun2id.items()}

    with open(cache_path, "wb") as f:
        pickle.dump({'noun2id': noun2id, 'id2noun': id2noun}, f)
    print("✅ 명사 사전 저장 완료")
    return noun2id, id2noun

# -----------------------
# 명사 수 캐시 활용 (각 텍스트에서 추출된 명사의 개수)
# -----------------------
def build_or_load_noun_count_cache(texts, cache_path):
    if os.path.exists(cache_path):
        print(f"\n📦 명사 수 캐시 발견 → {cache_path} 로드 중...")
        with open(cache_path, "rb") as f:
            noun_counts = pickle.load(f)
        return noun_counts
    print(f"\n🆕 명사 수 캐시 없음 → 새로 생성 중... ({cache_path})")
    noun_counts = []
    for text in tqdm(texts, desc="🔍 명사 수 캐시 생성 중"):
        nouns = extract_nouns(text)
        noun_counts.append(len(nouns))
    with open(cache_path, "wb") as f:
        pickle.dump(noun_counts, f)
    print("✅ 명사 수 캐시 저장 완료")
    return noun_counts

# -----------------------
# 명사 3개 조합 추출 함수 (GPU 사용, 단 과도한 명사 수일 경우 CPU 방식으로 전환)
# -----------------------
def extract_triplets(text, noun2id, cpu_threshold=50):
    """
    텍스트에서 명사를 추출한 후, 명사 ID를 이용해 3개 조합(triplet)을 생성합니다.
    명사의 개수가 cpu_threshold보다 크면 CPU 기반 조합 생성으로 전환합니다.
    """
    if not isinstance(text, str):
        return []
    nouns = extract_nouns(text)
    ids = [noun2id[n] for n in nouns if n in noun2id]
    ids = sorted(ids)
    if len(ids) < 3:
        return []
    # 명사 수가 많으면 CPU 방식 (itertools)로 처리하여 메모리 부담 완화
    if len(ids) > cpu_threshold:
        return list(itertools.combinations(ids, 3))
    # GPU 방식: cupy를 이용한 메쉬그리드 생성 (메모리 부담이 적당한 경우)
    cp_ids = cp.array(ids)
    n = cp_ids.shape[0]
    indices = cp.arange(n)
    i, j, k = cp.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i < j) & (j < k)
    triplet_indices = cp.stack([i[mask], j[mask], k[mask]], axis=-1)
    comb_gpu = cp_ids[triplet_indices]
    comb_list = [tuple(map(int, row)) for row in cp.asnumpy(comb_gpu)]
    return comb_list

# -----------------------
# 동적 배치 구성 함수
# -----------------------
def dynamic_batching(texts, noun_counts, max_combinations=1_000_000):
    """
    각 텍스트의 명사 수에 따라 예상 조합 수(nC3)를 계산한 후,
    누적 예상 조합 수가 max_combinations를 넘지 않는 범위 내에서 배치를 구성합니다.
    """
    batches = []
    current_batch = []
    current_total = 0
    for text, count in zip(texts, noun_counts):
        if count < 3:
            estimated_combs = 0
        else:
            estimated_combs = (count * (count - 1) * (count - 2)) // 6
        # 만약 한 텍스트가 단독으로 임계값을 초과하면 별도 배치로 처리
        if estimated_combs > max_combinations:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_total = 0
            batches.append([text])
            continue
        # 현재 배치에 추가 시 임계값을 초과하면 현재 배치를 분리
        if current_total + estimated_combs > max_combinations and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_total = 0
        current_batch.append(text)
        current_total += estimated_combs
    if current_batch:
        batches.append(current_batch)
    return batches

# -----------------------
# Counter를 디스크에 분산 저장
# -----------------------
def save_partial_counter(counter, prefix, batch_id):
    fname = os.path.join(CHECKPOINT_DIR, f"{prefix}_partial_{batch_id}.pkl.gz")
    with gzip.open(fname, "wb") as f:
        pickle.dump(counter, f)

# -----------------------
# 부분 체크포인트 병합 함수
# -----------------------
def merge_partial_checkpoints(prefix, file_pattern_prefix, merge_threshold=100):
    """
    CHECKPOINT_DIR 내에 file_pattern_prefix로 시작하는 파일이 merge_threshold 이상 쌓이면,
    해당 파일들을 병합한 후 하나의 파일로 저장하고, 원본 파일들을 삭제합니다.
    """
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) 
                    if f.startswith(file_pattern_prefix) and f.endswith('.pkl.gz')])
    if len(files) >= merge_threshold:
        print(f"Merging {len(files)} checkpoint files for {prefix}...")
        total_counter = Counter()
        for fname in files:
            full_path = os.path.join(CHECKPOINT_DIR, fname)
            with gzip.open(full_path, 'rb') as f:
                part = pickle.load(f)
                total_counter.update(part)
            os.remove(full_path)
        merged_filename = os.path.join(CHECKPOINT_DIR, f"{prefix}_merged.pkl.gz")
        with gzip.open(merged_filename, "wb") as f:
            pickle.dump(total_counter, f)
        print(f"Merged file saved as {merged_filename}")
        return total_counter
    return None

# -----------------------
# 분산된 Counter 병합 함수 (최종 결과)
# -----------------------
def merge_all_counters(prefix):
    total = Counter()
    # partial 파일과 병합된 파일 모두 포함
    files = [f for f in os.listdir(CHECKPOINT_DIR) 
             if (f.startswith(f"{prefix}_partial_") or f.startswith(f"{prefix}_merged_")) and f.endswith('.pkl.gz')]
    for fname in sorted(files):
        full_path = os.path.join(CHECKPOINT_DIR, fname)
        with gzip.open(full_path, 'rb') as f:
            part = pickle.load(f)
            total.update(part)
        os.remove(full_path)
    return total

# -----------------------
# 워커 초기화 함수 (multiprocessing)
# -----------------------
def init_worker(noun2id_arg):
    global noun2id_global, tagger
    noun2id_global = noun2id_arg
    # 각 워커에서 별도의 태거 인스턴스 초기화
    tagger = MeCab.Tagger()

def process_text_wrapper(text):
    return extract_triplets(text, noun2id_global)

# -----------------------
# 동적 배치 기반 체크포인트 병렬 처리 함수
# -----------------------
def extract_with_checkpoint_parallel_dynamic(batches, filename_prefix, noun2id, num_workers=4, merge_threshold=100):
    tmp_file = os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_progress.pkl")
    start_batch_idx = 0
    if os.path.exists(tmp_file):
        with open(tmp_file, "rb") as f:
            saved = pickle.load(f)
            start_batch_idx = saved.get("last_batch", -1) + 1
        print(f"\n🔄 체크포인트 로드됨: 배치 {start_batch_idx}부터 시작")
    total_batches = len(batches)
    pbar = tqdm(total=total_batches, initial=start_batch_idx, desc=filename_prefix)
    try:
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(noun2id,)) as pool:
            for batch_idx in range(start_batch_idx, total_batches):
                batch_texts = batches[batch_idx]
                results = pool.imap_unordered(process_text_wrapper, batch_texts)
                batch_counter = Counter()
                for triplets in tqdm(results, total=len(batch_texts), desc=f"Batch {batch_idx} 진행", leave=False):
                    batch_counter.update(triplets)
                save_partial_counter(batch_counter, filename_prefix, batch_idx)
                with open(tmp_file, "wb") as f:
                    pickle.dump({"last_batch": batch_idx}, f)
                
                # 주기적으로 부분 체크포인트 파일 병합
                merge_partial_checkpoints(filename_prefix, f"{filename_prefix}_partial_", merge_threshold)
                
                pbar.update(1)
    finally:
        pbar.close()
        # 진행 상태 파일 삭제 여부: 필요에 따라 아래를 주석 처리하거나 해제
        # if os.path.exists(tmp_file):
        #     os.remove(tmp_file)
    return merge_all_counters(filename_prefix)

# -----------------------
# 메인 로직
# -----------------------
def analyze_keyword_triplets(csv_path):
    df = read_csv_with_encoding(csv_path)

    # 'body' 컬럼 기준으로 요약 있음/없음 분리
    with_summary = df[df['summary'].notnull() & (df['summary'].str.strip() != '')]['body'].dropna().tolist()
    without_summary = df[df['summary'].isnull() | (df['summary'].str.strip() == '')]['body'].dropna().tolist()

    print(f"요약 있음 기사 수: {len(with_summary)}")
    print(f"요약 없음 기사 수: {len(without_summary)}")

    all_texts = with_summary + without_summary
    noun2id, id2noun = build_or_load_noun_dictionary(all_texts)

    # 명사 수 캐시 생성 (with_summary와 without_summary 각각)
    with_count_cache_path = "noun_count_cache_with.pkl"
    without_count_cache_path = "noun_count_cache_without.pkl"
    with_noun_counts = build_or_load_noun_count_cache(with_summary, with_count_cache_path)
    without_noun_counts = build_or_load_noun_count_cache(without_summary, without_count_cache_path)

    # 동적 배치 구성 (max_combinations 값은 테스트 환경에 맞게 조절)
    max_combinations = 1_000_000
    with_batches = dynamic_batching(with_summary, with_noun_counts, max_combinations)
    without_batches = dynamic_batching(without_summary, without_noun_counts, max_combinations)

    print(f"\nwith_summary: 동적 배치 개수 = {len(with_batches)}")
    print(f"without_summary: 동적 배치 개수 = {len(without_batches)}")

    # 동적 배치별로 병렬 처리
    with_counter = extract_with_checkpoint_parallel_dynamic(with_batches, "with_summary", noun2id, num_workers=4, merge_threshold=100)
    without_counter = extract_with_checkpoint_parallel_dynamic(without_batches, "without_summary", noun2id, num_workers=4, merge_threshold=100)

    # 요약 없음 뉴스에만 등장하는 3개 조합 통계 산출
    full_triplet_stats = []
    for triplet, without_count in without_counter.items():
        with_count = with_counter.get(triplet, 0)
        if with_count == 0 and without_count >= 3:
            keywords = [id2noun[i] for i in triplet]
            full_triplet_stats.append({
                "keyword1": keywords[0],
                "keyword2": keywords[1],
                "keyword3": keywords[2],
                "without_summary_count": without_count,
                "with_summary_count": with_count,
                "frequency_gap": without_count - with_count
            })

    full_triplet_stats = sorted(full_triplet_stats, key=lambda x: x["without_summary_count"], reverse=True)

    print("\n🧩 요약 없음 뉴스에만 등장하는 키워드 3개 조합 Top 30:")
    for item in full_triplet_stats[:30]:
        print(f"{item['keyword1']}, {item['keyword2']}, {item['keyword3']}: {item['without_summary_count']}")

    df_result = pd.DataFrame(full_triplet_stats)
    result_csv = "keyword_triplet_stats.csv"
    df_result.to_csv(result_csv, index=False, encoding='utf-8-sig')
    print(f"\n📁 결과 저장됨 → {result_csv}")

# -----------------------
# 추가: CSV 후처리 (필터링) 함수
# -----------------------
def filter_triplets_csv(input_csv_path, output_csv_path):
    import pandas as pd
    # tqdm의 notebook 버전을 사용 (노트북 환경이라면)
    from tqdm.notebook import tqdm

    # CSV 읽기
    df = pd.read_csv(input_csv_path)

    # 제거할 키워드 목록 (완전 일치)
    exact_keywords_to_remove = ['제보', '카카오톡', '연합뉴스', '뉴스','저작권자', '송고', '배포', '금지', '기자', '학습','무단', '삼성전자', '이투데이','한겨례',
                            '반도체','메모리','제조','공급','투자','산업','전환','확대','고객','실적','전망','확인','트럼프','행정부','우려','지속','제공',
                            '능력','세대','센터','데이터','첨단', '격차', '성장', '경영', '공급', '기업', '부족','그룹','사업', '개발','데일리','공정','제조',
                            '설비', '글로벌','쇼크', '경제', '한국','과학','아시아','미래','주도', '운영','가능','한겨레','감소','시장','혁신','중국','기회','미국',
                            '경쟁력','AI','인공지능','인도','수출','수입','영업','극복','강화','파운드리']
    # 제거할 키워드 목록 (부분 포함)
    partial_keywords_to_remove = ['일보', '신문','삼성','한국','전자']

    tqdm.pandas(desc="Filtering rows (fast)")

    # 1. 완전 일치하는 경우 제거: 각 셀의 값이 exact_keywords_to_remove에 완전히 일치하면 해당 행 제거
    mask_exact = df[['keyword1', 'keyword2', 'keyword3']].progress_apply(
        lambda row: not any(word in exact_keywords_to_remove for word in row),
        axis=1
    )
    df_exact_filtered = df[mask_exact].reset_index(drop=True)

    # 2. 일부라도 포함되면 제거: 각 셀의 값 내에 partial_keywords_to_remove에 포함된 단어가 있으면 해당 행 제거
    mask_partial = df_exact_filtered[['keyword1', 'keyword2', 'keyword3']].progress_apply(
        lambda row: not any(kw in word for word in row for kw in partial_keywords_to_remove),
        axis=1
    )
    filtered_df = df_exact_filtered[mask_partial].reset_index(drop=True)

    print("\n🧹 필터링 완료된 결과:")
    print(filtered_df)
    filtered_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📁 최종 결과 저장됨 → {output_csv_path}")

# -----------------------
# 메인 진입점
# -----------------------
if __name__ == '__main__':
    # 플랫폼에 따라 적절한 start method 선택 (Unix는 'spawn' 사용 권장)
    if os.name == 'posix':
        multiprocessing.set_start_method('spawn', force=True)
    else:
        multiprocessing.set_start_method('spawn', force=True)
    
    # 원본 CSV 파일 경로 (예시)
    csv_path = "/home/kororu/KoBERT/articles_summary24.csv"
    analyze_keyword_triplets(csv_path)
    
    # 분석 후 생성된 keyword_triplet_stats.csv 파일을 후처리하여 필터링 실행
    filter_triplets_csv("keyword_triplet_stats.csv", "3keywords.csv")
