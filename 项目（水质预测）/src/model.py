# -*- coding: utf-8 -*-
"""
基于 PyTorch 的多变量 LSTM：
- 输入形状: (batch, seq_len, num_features)
- 输出形状: (batch, num_targets) 使用最后时刻隐藏状态经过全连接映射
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math


class LSTMForecaster(nn.Module):
    """多变量到多目标的 LSTM 预测模型。

    增强：增加可选池化方式以汇聚时间维信息（默认与历史实现一致，取最后时刻）。
    - pooling="last": 取最后一个时间步隐藏状态（保持原行为）
    - pooling="mean": 对时间维做平均池化
    - pooling="attn": 基于可学习注意力对时间维加权求和
    """

    def __init__(
        self,
        num_features: int,
        num_targets: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        # 池化方式校验（默认保持 "last"）
        pooling = (pooling or "last").lower()
        assert pooling in {"last", "mean", "attn"}, "pooling 必须为 'last' | 'mean' | 'attn'"
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 注意力层：当选择 attn 池化时启用
        if self.pooling == "attn":
            # 使用一个单层感知器对每个时间步打分，随后做 softmax 得到权重
            self.attn_score = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)
        # 根据池化方式，将时间维 (T) 聚合为单个隐藏向量 (B, H)
        if self.pooling == "last":
            # 兼容原行为：仅使用最后一个时间步
            feat = out[:, -1, :]
        elif self.pooling == "mean":
            # 对时间维做平均池化
            feat = out.mean(dim=1)
        else:  # attn
            # 基于可学习注意力的加权汇聚
            # scores: (B, T, 1) -> weights: (B, T, 1)
            scores = self.attn_score(out)
            weights = torch.softmax(scores, dim=1)
            feat = (weights * out).sum(dim=1)
        y = self.head(feat)
        return y



class SinusoidalPositionalEncoding(nn.Module):
    """正余弦位置编码（支持 batch_first）。"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerForecaster(nn.Module):
    """基于 TransformerEncoder 的多变量到多目标预测模型。

    - 输入: (B, T, F)
    - 先线性投影到 d_model，叠加位置编码
    - 通过 TransformerEncoder 编码，随后按 pooling 聚合时间维
    - 经过 MLP 头得到 (B, num_targets)
    """

    def __init__(
        self,
        num_features: int,
        num_targets: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        pooling: str = "last",
        max_len: int = 1000,
    ) -> None:
        super().__init__()
        pooling = (pooling or "last").lower()
        assert pooling in {"last", "mean", "attn"}, "pooling 必须为 'last' | 'mean' | 'attn'"
        self.pooling = pooling
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        if self.pooling == "attn":
            self.attn_score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        out = self.encoder(h)  # (B, T, D)
        if self.pooling == "last":
            feat = out[:, -1, :]
        elif self.pooling == "mean":
            feat = out.mean(dim=1)
        else:
            scores = self.attn_score(out)
            weights = torch.softmax(scores, dim=1)
            feat = (weights * out).sum(dim=1)
        y = self.head(feat)
        return y

