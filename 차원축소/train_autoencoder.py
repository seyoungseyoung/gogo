# train_autoencoder.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

#############################
# 1. Hyperparameters
#############################
def get_hparams():
    """
    Returns a dictionary of hyperparameters for consistent usage
    in both training and shrink (encoding) phases.
    """
    return {
        "HUGGINGFACE_MODEL_NAME": "",
        "INPUT_DIM": 1536,
        # "Deep and Great" network layers
        "HIDDEN_DIMS_ENCODER": [1024, 512, 256],
        "HIDDEN_DIMS_DECODER": [256, 512, 1024],
        "BOTTLENECK_DIM": 128,
        "EPOCHS": 200,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 1e-4,
        # Where to save/load the final model
        "MODEL_SAVE_PATH": "C:/Users/tpdud/code/gogo/차원축소/autoencoder.pth"
    }

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

#############################
# 2. Encoder / Decoder
#############################
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super(Encoder, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            # Using GELU with approximate='tanh'
            layers.append(nn.GELU(approximate='tanh'))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, bottleneck_dim))
        # Final layer also with GELU (optional)
        layers.append(nn.GELU(approximate='tanh'))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, bottleneck_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        layers = []
        in_dim = bottleneck_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU(approximate='tanh'))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

#############################
# 3. AutoEncoder model
#############################
class AutoEncoder(nn.Module):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(
            input_dim=hparams["INPUT_DIM"],
            hidden_dims=hparams["HIDDEN_DIMS_ENCODER"],
            bottleneck_dim=hparams["BOTTLENECK_DIM"]
        )
        self.decoder = Decoder(
            bottleneck_dim=hparams["BOTTLENECK_DIM"],
            hidden_dims=hparams["HIDDEN_DIMS_DECODER"],
            output_dim=hparams["INPUT_DIM"]
        )

        if hparams["HUGGINGFACE_MODEL_NAME"]:
            print(f"Loading huggingface model: {hparams['HUGGINGFACE_MODEL_NAME']}")
            # from transformers import AutoModel
            # hf_model = AutoModel.from_pretrained(hparams["HUGGINGFACE_MODEL_NAME"])
            # If needed, partially initialize with hf_model weights

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

#############################
# 4. Dataset & collate_fn
#############################
class EmbeddingDataset(Dataset):
    def __init__(self, json_path, input_dim):
        """
        Expects JSON with columns ['ID','embedding'].
        'ID' is forced to string to preserve underscores.
        """
        df = pd.read_json(json_path)
        # Force ID to string to keep underscores (e.g. "20200101_1")
        df['ID'] = df['ID'].astype(str)

        valid_embeddings = []
        invalid_count = 0

        for _, row in df.iterrows():
            emb = row['Embedding']
            if emb is None or len(emb) != input_dim:
                invalid_count += 1
            else:
                valid_embeddings.append(emb)
        
        self.embeddings = np.array(valid_embeddings, dtype=np.float32)
        
        print(f"[{json_path}] Dropped embeddings: {invalid_count}")
        print(f"[{json_path}] Final used embeddings shape: {self.embeddings.shape}")

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]

def collate_fn(batch):
    # Convert list-of-arrays to (batch_size, input_dim) in one shot
    batch_np = np.stack(batch, axis=0)
    return torch.tensor(batch_np, dtype=torch.float32)

#############################
# 5. Main Training Function
#############################
def train_autoencoder(data_paths):
    hparams = get_hparams()
    device = get_device()

    # 1) Load & concatenate data
    datasets = []
    for path in data_paths:
        ds = EmbeddingDataset(path, input_dim=hparams["INPUT_DIM"])
        datasets.append(ds)

    all_embeddings = np.concatenate([d.embeddings for d in datasets], axis=0)
    print("Total embeddings shape:", all_embeddings.shape)

    class ConcatEmbDataset(Dataset):
        def __init__(self, emb_array):
            self.emb_array = emb_array
        def __len__(self):
            return len(self.emb_array)
        def __getitem__(self, idx):
            return self.emb_array[idx]

    concat_ds = ConcatEmbDataset(all_embeddings)
    loader = DataLoader(
        concat_ds,
        batch_size=hparams["BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2) Initialize model
    model = AutoEncoder(hparams).to(device)
    model.train()

    # 3) Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=hparams["LEARNING_RATE"])
    criterion = nn.MSELoss()

    # 4) Single progress bar for epochs (no per-batch bar)
    epochs_bar = tqdm(range(hparams["EPOCHS"]), desc="Training", ncols=80)
    for epoch_idx in epochs_bar:
        epoch_loss = 0.0

        for batch_data in loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            recon = model(batch_data)
            loss = criterion(recon, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        # Update tqdm bar in one line
        epochs_bar.set_postfix({"epoch": epoch_idx+1, "loss": f"{avg_loss:.6f}"})

    # 5) Save model
    torch.save(model.state_dict(), hparams["MODEL_SAVE_PATH"])
    print(f"\nAutoEncoder model saved to {hparams['MODEL_SAVE_PATH']}")

#############################
# 6. Script Entry
#############################
if __name__ == "__main__":
    data_file_list = [
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
        "C:/Users/tpdud/code/gogo/Database/summary/전처리 후/checkpoint_005930_2019_filtered.json",
    ]
    train_autoencoder(data_file_list)