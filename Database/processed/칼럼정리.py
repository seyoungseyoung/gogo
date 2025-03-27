import pandas as pd
import os

# 기본 경로
base_path = r"C:\Users\tpdud\code\gogo\Database\raw"

# 연도 범위
years = range(2019, 2026)

# 필요한 컬럼
columns_to_keep = ['Title', 'Date', 'Link', 'Press', 'Body','ID']

# 실행 디렉토리 기준으로 저장
output_dir = os.getcwd()

for year in years:
    file_name = f"005930_{year}_processed.csv"
    input_path = os.path.join(base_path, file_name)

    try:
        # CSV 파일 읽기
        df = pd.read_csv(input_path)

        # 필요한 컬럼만 추출
        df_filtered = df[columns_to_keep]

        # 저장 경로 설정
        output_file = os.path.join(output_dir, f"005930_{year}_filtered.csv")

        # 저장
        df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"{year}년 데이터 저장 완료: {output_file}")

    except FileNotFoundError:
        print(f"{year}년 파일이 없습니다: {input_path}")
    except Exception as e:
        print(f"{year}년 파일 처리 중 오류 발생: {e}")
