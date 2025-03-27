import os
import json

embedding_path = r"C:\Users\tpdud\code\gogo\Database\embedding"
file_list = [os.path.join(embedding_path, file) for file in os.listdir(embedding_path) if file.startswith("checkpoint_") and file.endswith(".json")]

for file in file_list:
    with open(file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            print(f"File: {file}")
            print("Type:", type(data))
            if isinstance(data, list) and len(data) > 0:
                print("First item type:", type(data[0]))
                print("First item:", data[0])
            else:
                print("Content:", data)
            print("-" * 40)
        except Exception as e:
            print(f"Failed to load JSON from {file}: {e}")
