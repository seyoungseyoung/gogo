from merge_utils import *
import pandas as pd
from datetime import datetime

# Main Execution
# Main Execution
if __name__ == "__main__":
    version = "shrink_128"
    embedding_dim = 128
    file_end = "shrink_128.json"   # 🔁 바뀐 부분
    
    processed_path = "C:/Users/tpdud/code/gogo/Database/summary"
    embedding_path = "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/shrink"
    financial_path = "C:/Users/tpdud/code/Byte_재우/Database/Local/Final/samsung_only_scaled.csv"
    output_path = f"C:/Users/tpdud/code/gogo/merge/final_{version}.csv"
    fill_record_path = f"C:/Users/tpdud/code/gogo/merge/fill_records_{version}.csv"

    embeddings = load_embeddings(embedding_path, file_end)
    processed_df = load_processed_data(processed_path, embeddings)
    min_date = datetime(2019, 1, 1)
    selected_ids, fill_records = select_top_ids(processed_df, min_date)
    fill_records_df = pd.DataFrame([{"Date": date, "Sources": rec["sources"], "IDs": rec["ids"]} for date, rec in fill_records.items()])
    fill_records_df.to_csv(fill_record_path, index=False)
    news_data = match_embeddings(selected_ids, embeddings, embedding_dim)
    merge_with_financial(financial_path, news_data, output_path, embedding_dim)