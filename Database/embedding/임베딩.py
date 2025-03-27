from utils import *
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm

def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '-o', '--output_dir', 
        type=str, 
        required=False, 
        default=r"C:\Users\tpdud\code\gogo\Database\embedding"
    )
    parser.add_argument(
        '-y', '--year', 
        type=str, 
        required=True, 
        help="Year or year range (e.g., 2019 or 2019-2025)"
    )
    return parser

def process_year(year: str, output_dir: str):
    # 입력 파일 경로: 뉴스키워드전처리 폴더의 keep_{year}.csv 파일 사용
    input_file = fr"C:\Users\tpdud\code\gogo\Database\뉴스키워드전처리\keep_{year}.csv"
    print(f"Processing file: {input_file}")
    
    load_env()  # .env1 파일에서 환경변수 로드
    df = load_data(input_file)
    
    # 배치 JSON 파일 경로를 output_dir 하위에 저장
    batch_json_path = os.path.join(output_dir, f"batches_info_{year}.json")
    if not os.path.exists(batch_json_path):
        print("Batch JSON not found. Creating new batch JSON...")
        make_batch(df, output_dir, batch_size=20, year=year)
    else:
        print("Batch JSON file exists. Loading batch info...")
    with open(batch_json_path, "r", encoding="utf-8") as f:
        batch_info = json.load(f)

    # 체크포인트 파일 경로 (중단 후 재시작용)
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{year}.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            embeddings_result = json.load(f)
        print("Checkpoint file loaded. Resuming from checkpoint...")
    else:
        embeddings_result = []
        print("No checkpoint file found. Starting new embedding process...")

    # 이미 처리된 ID 목록 구성
    processed_ids = set(item["ID"] for item in embeddings_result)
    batch_keys = sorted([key for key in batch_info.keys() if key.startswith("Batch_")])
    
    # 각 배치에 대해 임베딩 처리 수행
    for batch_key in tqdm(batch_keys, desc=f"Processing batches for {year}"):
        batch_ids = batch_info[batch_key]
        new_batch_ids = [bid for bid in batch_ids if bid not in processed_ids]
        if not new_batch_ids:
            continue

        batch_df = df[df["ID"].isin(new_batch_ids)]
        batch_result = process_embeddings(batch_df, model="text-embedding-3-small", max_tokens=7000)
        embeddings_result.extend(batch_result)
        processed_ids.update(new_batch_ids)
        
        # 배치 처리 후 체크포인트 저장
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(embeddings_result, f, ensure_ascii=False, indent=4)

    # 실패한 임베딩 재처리
    print("Retrying failed embeddings...")
    retry_failed_embeddings(checkpoint_file, df, model="text-embedding-3-small", max_tokens=7000)
    print(f"Embedding processing complete for year {year}.")

def main():
    args = create_parser().parse_args()
    output_dir = args.output_dir
    year_arg = args.year

    # 단일 연도 또는 연도 범위 처리
    if '-' in year_arg:
        start_year, end_year = year_arg.split('-')
        years = [str(y) for y in range(int(start_year), int(end_year) + 1)]
    else:
        years = [year_arg]

    for yr in years:
        print(f"\n=== Starting processing for year {yr} ===")
        process_year(yr, output_dir)

if __name__ == "__main__":
    main()
