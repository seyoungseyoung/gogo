import torch
import torch.nn as nn
import math
from torch_utils import CustomTransformerEncoderLayer, CustomTransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerForecast(nn.Module):
    def __init__(self, input_size, d_model, num_layers, input_steps, news_dim=640, nhead=4, output_size=1, dropout=0.1):
        """
        Transformer 기반 시계열 예측 모델.

        :param input_size: X_nonnews의 feature 차원 (예: 5개 변수)
        :param d_model: Transformer의 hidden dimension
        :param num_layers: Transformer Encoder layer 수
        :param input_steps: 입력 시퀀스 길이 (예: 40)
        :param news_dim: 뉴스 벡터 차원 (예: 640)
        """
        super(TransformerForecast, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        news_in_dim = news_dim * input_steps

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            news_in_dim=news_in_dim,
            batch_first=True
        )
        self.transformer_encoder = CustomTransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x_nonnews, x_news):
        # x_nonnews: (batch, seq_len, input_size)
        # x_news: (batch, seq_len, news_dim)

        x_nonnews = self.input_projection(x_nonnews)  # (batch, seq_len, d_model)
        x_nonnews = self.pos_encoder(x_nonnews)       # (batch, seq_len, d_model)

        batch_size = x_news.size(0)
        x_news = x_news.view(batch_size, -1)          # (batch, seq_len * news_dim)

        transformer_out = self.transformer_encoder(x_nonnews, x_news)  # (batch, seq_len, d_model)
        out = transformer_out[:, -1, :]                # 마지막 시점 (batch, d_model)
        out = self.fc(out)                             # (batch, output_size)
        return out.squeeze(-1)                         # (batch,)
