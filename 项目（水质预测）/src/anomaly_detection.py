# -*- coding: utf-8 -*-
"""
异常检测与处理：
1) 中位数绝对偏差（MAD）法：适合稳健检测单变量异常。
2) 隔离森林（IsolationForest）：适合多变量综合异常。

输出策略：将异常值置为 NaN，交由后续插值/填补处理，保证数据连续性与稳定性。
"""
from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies_mad(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """基于 MAD 的单变量异常检测，将异常位置置为 NaN。

    参数：
        df: 输入数据（数值列）
        columns: 要检测的列名，为 None 则全部数值列
        threshold: 判定阈值，常用 3.5
    返回：
        新的 DataFrame，异常位置已置为 NaN
    """
    result = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        series = df[col].astype(float)
        median = np.nanmedian(series)
        mad = np.nanmedian(np.abs(series - median)) + 1e-9
        modified_z = 0.6745 * (series - median) / mad
        mask = np.abs(modified_z) > threshold
        result.loc[mask, col] = np.nan
    return result


def detect_anomalies_isoforest(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """基于 IsolationForest 的多变量异常检测，将异常位置置为 NaN。
    注意：会综合列进行检测，若判断某行整体为异常，则对指定列置 NaN。
    """
    result = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[columns].astype(float).values
    # 对缺失进行临时简单填补，避免模型失败（后续会再统一插值）
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)  # -1 表示异常
    mask = labels == -1
    result.loc[mask, columns] = np.nan
    return result


