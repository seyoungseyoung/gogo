import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
        # For the MLP on X_news:
        news_in_dim: int = 640,  # e.g., 1280 from your description
    ):
        """
        A custom Transformer-like encoder layer that:
          - takes one 'normal' Transformer input (X_nonnews)
          - takes an auxiliary input (X_news)
          - merges them after the feed-forward network.
        """

        super().__init__()

        # ----------------------------
        # 1) Standard Self-Attention
        # ----------------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # ----------------------------
        # 2) Feed-Forward Network
        # ----------------------------
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # ----------------------------
        # 3) LayerNorms, Dropouts
        # ----------------------------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation.lower() == "relu":
            self.activation_fn = F.relu
        elif activation.lower() == "gelu":
            self.activation_fn = F.gelu
        else:
            raise ValueError("Only 'relu' or 'gelu' supported here for simplicity.")

        # ------------------------------------------------
        # 4) MLP to process X_news (1280 -> 768 -> 384 -> d_model)
        ### 지금은 15360 -> 1536 -> 768 -> 384 -> d_model 으로 변경
        # ------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(news_in_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, d_model),
            # nn.ReLU(),
            # nn.Linear(384, d_model),
        )


    def forward(
        self,
        X_nonnews: Tensor,       # shape [batch_size, seq_len, d_model] (if batch_first=True)
        X_news: Tensor,          # shape [batch_size, news_in_dim] or [batch_size, ..., 1280]
        src_mask: Tensor = None,
        src_key_padding_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
          X_nonnews: The usual Transformer input
          X_news:    The auxiliary 'news' features
        """

        # ------------------------------------------------------------------
        # 1) Self-Attention block (with standard residual connection)
        # ------------------------------------------------------------------
        # MultiheadAttention expects (B, T, D) if batch_first=True
        attn_out, _ = self.self_attn(
            X_nonnews,
            X_nonnews,
            X_nonnews,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        
        mlp_out = self.mlp(X_news)  # [batch, d_model]
        mlp_out = mlp_out.unsqueeze(1).repeat(1, X_nonnews.size(1), 1)  # → [batch, seq_len, d_model]

        x = mlp_out + self.dropout1(attn_out) + X_nonnews
        # Standard residual
        # x = X_nonnews + self.dropout1(attn_out)
        # x = mlp_out + self.dropout1(attn_out)
        x = mlp_out + self.dropout1(attn_out) + X_nonnews
        x = self.norm1(x)

        # ------------------------------------------------------------------
        # 2) Feed-Forward block (normally: x = x + FF(x))
        #    but we'll separate the "FF(x)" for clarity
        # ------------------------------------------------------------------
        ff_out = self.linear2(self.dropout(self.activation_fn(self.linear1(x))))
        ff_out = self.dropout2(ff_out)

        
        # x = ff_out + mlp_out
        x = ff_out + mlp_out + x

        # Then apply LayerNorm
        x = self.norm2(x)

        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: CustomTransformerEncoderLayer,
        num_layers: int,
        norm: nn.LayerNorm = None,
    ):
        """
        A stack of CustomTransformerEncoderLayers.
        
        Args:
            encoder_layer: A *configured* instance of CustomTransformerEncoderLayer
                           (will be deep-copied num_layers times).
            num_layers:    How many times to stack this layer.
            norm:          Optional final LayerNorm applied to the output.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm  # optional final LayerNorm

    def forward(
        self,
        X_nonnews: Tensor,
        X_news: Tensor,
        mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        """
        Pass the inputs (X_nonnews, X_news) through each CustomTransformerEncoderLayer in turn.
        """
        output = X_nonnews
        for layer in self.layers:
            output = layer(
                X_nonnews=output,
                X_news=X_news,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output