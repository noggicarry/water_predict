# -*- coding: utf-8 -*-
"""
数据预处理：
- 读取原始 CSV，按配置频率重采样
- 异常检测（MAD/IsolationForest）置 NaN
- 缺失值插值与前后填补
- 标准化与滑动窗口样本构造
"""
from __future__ import annotations

from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    RAW_DATA_FILE,
    PROCESSED_DIR,
    TIME_COL,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    FREQ,
    LOOKBACK_STEPS,
    TRAIN_RATIO,
    VAL_RATIO,
    SCALER_FILE,
    INCLUDE_TIME_FEATURES,
)
from .utils import save_pickle, build_sliding_windows
from .anomaly_detection import detect_anomalies_mad


def load_and_resample() -> pd.DataFrame:
    """读取原始数据并按频率重采样，确保列顺序与类型。"""
    df = pd.read_csv(RAW_DATA_FILE)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).set_index(TIME_COL)
    # 仅保留需要的列
    df = df[FEATURE_COLUMNS]
    # 重采样为配置频率
    df = df.resample(FREQ).mean()
    return df


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """异常置 NaN + 缺失插值（线性）+ 前后填补，返回干净数据。"""
    # 先用 MAD 检测异常
    df2 = detect_anomalies_mad(df, columns=FEATURE_COLUMNS, threshold=3.5)
    # 插值与填补
    df2 = df2.interpolate(method="linear", limit_direction="both")
    # 使用 ffill/bfill 替代已弃用的 fillna(method=...)
    df2 = df2.ffill().bfill()
    return df2
def add_time_features(df: pd.DataFrame, freq: str) -> Tuple[pd.DataFrame, List[str]]:
    """根据频率添加时间正余弦特征，返回新增列名。

    - 'D': 添加 day_of_week 与 month 的正余弦
    - 'H': 添加 hour_of_day 与 day_of_week 的正余弦
    其他频率默认按 'D' 处理。
    """
    added: List[str] = []
    idx = df.index
    base = pd.DataFrame(index=idx)
    if freq.upper().startswith("H"):
        hour = idx.hour
        dow = idx.dayofweek
        base["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        base["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        base["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        base["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        added = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        # 默认按天
        dow = idx.dayofweek
        month = idx.month
        base["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        base["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        base["mon_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        base["mon_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
        added = ["dow_sin", "dow_cos", "mon_sin", "mon_cos"]
    # 追加到原数据（保持原 FEATURE_COLUMNS 在前）
    df2 = pd.concat([df, base[added]], axis=1)
    return df2, added



def split_train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按比例切分时间序列，保持顺序。"""
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """仅在训练集上拟合标准化器。"""
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    return scaler


def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """使用给定标准化器进行变换，保持列名。"""
    arr = scaler.transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def prepare_datasets() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """
    预处理完整流程，返回滑动窗口后的 X/y 训练、验证、测试集，以及列索引映射：
    - data_splits: {"train": (X, y), "val": (X, y), "test": (X, y)}
    - scaler 属性已落盘保存
    - col_index_map: {列名: 索引}
    """
    df_raw = load_and_resample()
    df_clean = clean_and_impute(df_raw)
    time_feature_cols: List[str] = []
    if INCLUDE_TIME_FEATURES:
        df_clean, time_feature_cols = add_time_features(df_clean, FREQ)

    # 保存清洗后数据
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DIR / "clean.csv", index=True)

    train_df, val_df, test_df = split_train_val_test(df_clean)
    scaler = fit_scaler(train_df)
    save_pickle(scaler, SCALER_FILE)

    train_scaled = transform_with_scaler(train_df, scaler)
    val_scaled = transform_with_scaler(val_df, scaler)
    test_scaled = transform_with_scaler(test_df, scaler)

    # 构造滑动窗口样本（单步预测训练）
    X_train, y_train_full = build_sliding_windows(train_scaled.values, LOOKBACK_STEPS, 1)
    X_val, y_val_full = build_sliding_windows(val_scaled.values, LOOKBACK_STEPS, 1)
    X_test, y_test_full = build_sliding_windows(test_scaled.values, LOOKBACK_STEPS, 1)

    # 选择目标列
    # 列索引映射应覆盖增广后的所有列
    all_columns: List[str] = list(train_scaled.columns)
    col_index_map = {c: i for i, c in enumerate(all_columns)}
    target_indices: List[int] = [col_index_map[c] for c in TARGET_COLUMNS]
    y_train = y_train_full[:, target_indices]
    y_val = y_val_full[:, target_indices]
    y_test = y_test_full[:, target_indices]

    data_splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    meta = {
        "num_features": len(all_columns),
        "num_targets": len(TARGET_COLUMNS),
        "all_columns": all_columns,
        "time_features": time_feature_cols,
    }
    return data_splits, meta, col_index_map


