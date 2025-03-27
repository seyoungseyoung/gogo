import json
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta

def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-w', '--window_size', type=int, required=True, help="윈도우 기간 (일수), 예: 30")
    parser.add_argument('-m', '--minimum_sample', type=int, required=True, help="최소 샘플 수, 예: 200")
    return parser

def main():
    args = create_parser().parse_args()
    window_size = args.window_size      # 예: 30일
    minimum_sample = args.minimum_sample  # 예: 200개

    # 입력 파일이 있는 경로 (체크포인트 파일)
    embedding_path = r"C:\Users\tpdud\code\gogo\Database\embedding"
    # "checkpoint_"로 시작하는 파일들만 선택
    file_dirs = [
        os.path.join(embedding_path, file)
        for file in os.listdir(embedding_path)
        if file.startswith("checkpoint_")
    ]

    # 각 파일을 로드하여 "ID" 컬럼만 저장
    file_data = {}
    for file in file_dirs:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_data[file] = [item["ID"] for item in data]

    start_date = datetime.strptime("20190101", "%Y%m%d")
    end_date = datetime.strptime("20250321", "%Y%m%d")

    results = []

    # 하루씩 이동하는 슬라이딩 윈도우
    current_date = start_date
    while current_date + timedelta(days=window_size - 1) <= end_date:
        print(f"Current date: {current_date}")
        # 윈도우에 포함되는 날짜 목록
        date_range = [current_date + timedelta(days=i) for i in range(window_size)]
        
        # 윈도우 시작 및 종료 날짜 문자열 (YYYYMMDD)
        window_start_str = current_date.strftime("%Y%m%d")
        window_end = current_date + timedelta(days=window_size - 1)
        window_end_str = window_end.strftime("%Y%m%d")
        
        # 윈도우에 포함된 날짜들의 연도 리스트 (중복 제거)
        year_list = list(set([str(d.year) for d in date_range]))
        
        # 연도 리스트에 해당하는 파일들을 매칭 (파일명에 연도 정보가 포함되어 있어야 함)
        matched_files = []
        for year in year_list:
            for f in file_dirs:
                if year in f:
                    matched_files.append(f)
        matched_files = list(set(matched_files))  # 중복 제거

        # 매칭된 파일들에서 윈도우 기간에 해당하는 ID만 수집
        all_ids = []
        for mf in matched_files:
            valid_ids = [id for id in file_data[mf] if window_start_str <= id[:8] <= window_end_str]
            all_ids.extend(valid_ids)

        # 최소 샘플 수 미달 시, 윈도우 시작일 이전의 파일들을 추가(backfill)
        back_offset = 1
        while len(all_ids) < minimum_sample:
            back_date = current_date - timedelta(days=back_offset)
            if back_date < datetime.strptime("20100101", "%Y%m%d"):
                break
            back_year = str(back_date.year)
            
            new_files = []
            for f in file_dirs:
                if back_year in f:
                    new_files.append(f)
            new_files = list(set(new_files))
            
            # 아직 매칭되지 않은 파일들을 추가
            for nf in new_files:
                if nf not in matched_files:
                    matched_files.append(nf)
                    valid_ids = [id for id in file_data[nf] if window_start_str <= id[:8] <= window_end_str]
                    all_ids.extend(valid_ids)
            
            back_offset += 1
            # 무한 루프 방지를 위해 최대 5년치로 제한
            if back_offset > 365 * 5:
                break

        result_dict = {
            window_end_str: [os.path.basename(x) for x in matched_files],
            "IDs": all_ids
        }
        results.append(result_dict)

        # 하루씩 슬라이딩 윈도우 이동
        current_date += timedelta(days=1)

    output_file = f"C:/Users/tpdud/code/gogo/클러스터/reference_{window_size}.json"
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)

    print(f"Done! Saved results to {output_file}")

if __name__ == "__main__":
    main()
