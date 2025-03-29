# shrink_embeddings.py

import json
import os
import numpy as np
import torch
import pandas as pd

from train_autoencoder import (
    get_hparams,
    AutoEncoder,
    get_device,
    EmbeddingDataset
)

def shrink_embeddings(input_json_path):
    """
    Loads the trained AutoEncoder, encodes (reduces) embeddings from input_json_path
    to the bottleneck dimension, and saves them to {original_filename}_shrink_{BOTTLENECK_DIM}.json
    """
    # 1) Same hyperparams used in training
    hparams = get_hparams()
    bottleneck = hparams["BOTTLENECK_DIM"]

    # 2) Load dataset
    dataset = EmbeddingDataset(input_json_path, input_dim=hparams["INPUT_DIM"])
    embeddings = dataset.embeddings  # shape (N, 1536)
    print(f"Loaded embeddings from {input_json_path}, shape: {embeddings.shape}")

    # 3) Load model
    model = AutoEncoder(hparams)  # same architecture
    device = get_device()
    model.load_state_dict(torch.load(hparams["MODEL_SAVE_PATH"], map_location=device))
    model.to(device)
    model.eval()
    print(f"AutoEncoder loaded from: {hparams['MODEL_SAVE_PATH']}")
    print(f"Bottleneck dimension: {bottleneck}")

    # 4) Encode
    with torch.no_grad():
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        z = model.encoder(emb_tensor)  # shape (N, bottleneck)
        z_cpu = z.cpu().numpy()

    # Convert to list-of-lists
    shrunk_list = z_cpu.tolist()

    # Match original IDs, ignoring invalid (dropped) ones
    df_original = pd.read_json(input_json_path)
    # Force ID to string to preserve underscores
    df_original['ID'] = df_original['ID'].astype(str)

    valid_data = []
    valid_idx = 0
    for i, row in df_original.iterrows():
        emb = row['embedding']
        if emb is not None and len(emb) == hparams["INPUT_DIM"]:
            # ID is already a string
            id_val = row['ID']
            record = {
                "ID": id_val,
                "embedding": shrunk_list[valid_idx]
            }
            valid_data.append(record)
            valid_idx += 1

    # 5) Save as {filename}_shrink_{bottleneck}.json
    base_name, ext = os.path.splitext(input_json_path)
    output_json_path = f"{base_name}_shrink_{bottleneck}{ext}"

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False)

    print(f"Shrunken embeddings saved to: {output_json_path}")
    print(f"Shape: {z_cpu.shape}")

if __name__ == "__main__":
    # Example usage for the 2020 file
    input_path = "../../../Database/Embedding/samsungE_2020_processed_embedded.json"
    shrink_embeddings(input_path)