import pandas as pd
import numpy as np
import ast

# 파일 로딩
df_ids = pd.read_csv('C:/Users/tpdud/code/gogo/merge/fill_records_shrink_128.csv')
df_sorted_list = pd.read_csv('C:/Users/tpdud/code/gogo/transformer/transformer_all_results (1).csv')
df_links_all = pd.concat([
    pd.read_csv(f'C:/Users/tpdud/code/gogo/Database/summary/005930_{year} (1).csv')
    for year in range(2019, 2026)
], ignore_index=True)

# 날짜 정제
df_sorted_list['Refined_Date'] = df_sorted_list['Date'].str.replace("-", "")
sorted_dataframe = df_sorted_list[['Refined_Date', 'Sorted_List']]
ids_dataframe = df_ids[['Date', 'IDs']]
links_dataframe = df_links_all[['Date', 'Title', 'Link', 'ID']].copy()

# ✅ ID 포맷 일치화: 뉴스 ID에서 _ 제거 (예: 20190306_016 → 20190306016)
links_dataframe['ID'] = links_dataframe['ID'].astype(str)
links_dataframe['ID_no_underscore'] = links_dataframe['ID'].str.replace('_', '', regex=False)

# 결과 저장 리스트
rearranged_ids_full_lst = []
link_lists = [[] for _ in range(5)]
title_lists = [[] for _ in range(5)]

# 본 처리 루프
for i in range(len(sorted_dataframe)):
    row = sorted_dataframe.iloc[i].copy()
    date = np.int64(row['Refined_Date'])

    row_ids = ids_dataframe[ids_dataframe['Date'] == date]
    if row_ids.empty:
        print(f"[SKIP] No IDs found for date: {date}")
        continue

    try:
        lst_ids = ast.literal_eval(row_ids['IDs'].squeeze())
        sorted_list = [i for i in ast.literal_eval(row['Sorted_List']) if 1 <= i <= len(lst_ids)][:5]
    except Exception as e:
        print(f"[ERROR] Failed to eval at row {i}: {e}")
        continue

    if len(sorted_list) < 5:
        print(f"[SKIP] Not enough valid indices at row {i}. Sorted_List: {sorted_list}")
        continue

    rearranged_ids = [lst_ids[idx - 1] for idx in sorted_list]

    links = []
    titles = []
    for id in rearranged_ids:
        id = str(id)  # ensure type match
        match = links_dataframe[links_dataframe['ID_no_underscore'] == id]
        link = match['Link'].squeeze() if not match.empty else "N/A"
        title = match['Title'].squeeze() if not match.empty else "N/A"
        links.append(link)
        titles.append(title)

    rearranged_ids_full_lst.append(rearranged_ids)

    for j in range(5):
        link_lists[j].append(links[j])
        title_lists[j].append(titles[j])

# 결과 저장
df_sorted_list = df_sorted_list.iloc[:len(rearranged_ids_full_lst)].copy()
df_sorted_list['Sorted IDs'] = rearranged_ids_full_lst

for i in range(5):
    df_sorted_list[f'Sorted Title {i+1}'] = title_lists[i]
    df_sorted_list[f'Sorted Links {i+1}'] = link_lists[i]

df_sorted_list.to_csv('sorted_links_TOP10_jeong(1).csv', index=False)
print("✅ 저장 완료: sorted_links_TOP10_jeong.csv")
