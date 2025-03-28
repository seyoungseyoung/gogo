import pandas as pd
import json
import numpy as np
import os

# 파일 경로 설정
cluster_json_path = r"C:\Users\tpdud\code\gogo\Database\Clusters\cluster_result.json"
input_csv_path = r"C:\Users\tpdud\code\gogo\Database\뉴스키워드전처리\keep_전체.csv"
output_csv_path = r"C:\Users\tpdud\code\gogo\Database\Clusters\keep_전체_clustered.csv"

# 클러스터링 결과 JSON 파일 로드
with open(cluster_json_path, 'r', encoding='utf-8') as f:
    cluster_data = json.load(f)

# ID -> 클러스터 라벨 매핑 딕셔너리 생성
id_to_cluster = {}
for entry in cluster_data:
    # entry 예: { "Date": "20190104", "Center_Link": [...], "Cluster_0": [...], "Cluster_1": [...], "Noise": [...] }
    for key, id_list in entry.items():
        if key in ["Date", "Center_Link"]:
            continue
        elif key == "Noise":
            for news_id in id_list:
                id_to_cluster[news_id] = "Noise"
        elif key.startswith("Cluster_"):
            # "Cluster_0" -> 0, "Cluster_1" -> 1, ...
            try:
                cluster_num = int(key.split("_")[1])
            except Exception as e:
                cluster_num = key  # 만약 숫자 변환이 안되면 그대로 사용
            for news_id in id_list:
                id_to_cluster[news_id] = cluster_num

# 기존 CSV 파일 로드 (ID 칼럼이 있어야 함)
df = pd.read_csv(input_csv_path)

# ID 칼럼을 기준으로 클러스터 라벨을 매핑, 없으면 NaN으로 남음
df['cluster'] = df['id'].map(id_to_cluster)

# 결과 CSV 저장 (UTF-8 with BOM 등 필요시 인코딩 조정)
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"클러스터 매핑 완료. 결과는 {output_csv_path}에 저장되었습니다.")
