# -*- coding: utf-8 -*-
"""
基于 PyTorch 的多变量 LSTM：
- 输入形状: (batch, seq_len, num_features)
- 输出形状: (batch, num_targets) 使用最后时刻隐藏状态经过全连接映射
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """多变量到多目标的 LSTM 预测模型。"""

    def __init__(
        self,
        num_features: int,
        num_targets: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
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
        last_hidden = out[:, -1, :]  # (B, H)
        y = self.head(last_hidden)
        return y


