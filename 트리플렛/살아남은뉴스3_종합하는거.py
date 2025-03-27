import pandas as pd

# CSV 파일 읽기
df1 = pd.read_csv("collected_articles_without_keyword_triplets.csv")
df2 = pd.read_csv("classified_articles_keep_extra.csv")

# 공통 칼럼 찾기
common_columns = list(set(df1.columns) & set(df2.columns))

# 공통 칼럼만 남기기
df1_common = df1[common_columns]
df2_common = df2[common_columns]

# 두 DataFrame concat
df_concat = pd.concat([df1_common, df2_common], ignore_index=True)

# Date 칼럼을 문자열로 변환 후, format 인자를 지정하여 datetime 타입으로 변환 ('20190101' 형식)
df_concat['Date'] = pd.to_datetime(df_concat['Date'].astype(str), format='%Y%m%d', errors='coerce')

# Date 칼럼에서 연도 추출
df_concat['Year'] = df_concat['Date'].dt.year

# 전체 파일 저장 (모든 데이터를 포함)
df_concat.to_csv("cleaned_all.csv", index=False)
print("cleaned_all.csv 파일 저장 완료")

# 각 연도별로 CSV 파일 저장
for year in df_concat['Year'].dropna().unique():
    df_year = df_concat[df_concat['Year'] == year]
    filename = f"cleaned_{int(year)}.csv"
    df_year.to_csv(filename, index=False)
    print(f"{filename} 파일 저장 완료")
