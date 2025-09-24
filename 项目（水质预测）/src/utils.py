# -*- coding: utf-8 -*-
"""
工具模块：随机种子、滑动窗口、评估指标、序列化工具等。
"""
from __future__ import annotations

import os
import random
import pickle
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    """设置随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_pickle(obj, file_path: Path) -> None:
    """保存 Python 对象至二进制文件。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path):
    """从二进制文件加载 Python 对象。"""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def build_sliding_windows(
    data: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于滑动窗口构造监督学习样本。

    参数：
        data: 形状 (T, D) 的时间序列数据（已完成对齐与填补）。
        lookback: 回看窗口长度。
        horizon: 预测步数（此处用于迭代预测，训练时为 1）。

    返回：
        X: (N, lookback, D)
        y: (N, D_target) 这里默认与特征同维或在外部选列
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total_length = data.shape[0]
    for end_idx in range(lookback, total_length - 1):
        start_idx = end_idx - lookback
        X_list.append(data[start_idx:end_idx, :])
        y_list.append(data[end_idx, :])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """计算 MAPE（百分比误差），避免除零。"""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 MAE。"""
    return float(np.mean(np.abs(y_true - y_pred)))


