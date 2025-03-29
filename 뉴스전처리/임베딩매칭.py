import pandas as pd
import json
import os

# 기본 경로
base_csv_path = r"C:\Users\tpdud\code\gogo\Database\뉴스키워드전처리\fuzz"
base_json_path = r"C:\Users\tpdud\code\gogo\Database\summary"
output_dir = os.path.join(base_json_path, "전처리 후")

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 연도별 반복
for year in range(2019, 2026):
    csv_file = os.path.join(base_csv_path, f"keep_{year}.csv")
    json_file = os.path.join(base_json_path, f"checkpoint_005930_{year}.json")
    output_file = os.path.join(output_dir, f"checkpoint_005930_{year}_filtered.json")

    # CSV 불러오기
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} 없음. 건너뜀.")
        continue

    df = pd.read_csv(csv_file)
    link_set = set(df['link'].dropna())

    # JSON 불러오기
    if not os.path.exists(json_file):
        print(f"❌ {json_file} 없음. 건너뜀.")
        continue

    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 필터링
    filtered_data = [item for item in json_data if item.get("Link") in link_set]

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"✅ {year}년 필터링 완료: {len(filtered_data)}개 저장됨 → {output_file}")
