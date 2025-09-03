# -*- coding: utf-8 -*-
"""
生成示例水质数据：包含 CO、NH3N、pH、DO、flow 等指标的日度数据，带有季节性、趋势、噪声，并随机注入缺失与异常点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import ensure_directories, RAW_DATA_FILE, TIME_COL


def generate_series(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    # 时间轴：生成 3 年日度数据
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n)

    # 季节性 + 趋势 + 噪声（仅示例，不代表真实物理规律）
    CO = 1.2 + 0.2 * np.sin(2 * np.pi * t / 365) + 0.001 * t + rng.normal(0, 0.05, n)
    NH3N = 0.6 + 0.15 * np.cos(2 * np.pi * t / 180) + 0.0008 * t + rng.normal(0, 0.03, n)
    pH = 7.0 + 0.1 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.02, n)
    DO = 6.5 + 0.5 * np.cos(2 * np.pi * t / 60) + rng.normal(0, 0.1, n)
    flow = 1.0 + 0.3 * np.sin(2 * np.pi * t / 90) + 0.0005 * t + rng.normal(0, 0.08, n)

    df = pd.DataFrame({
        TIME_COL: dates,
        "CO": CO,
        "NH3N": NH3N,
        "pH": pH,
        "DO": DO,
        "flow": flow,
    })

    # 注入缺失
    for col in ["CO", "NH3N", "pH", "DO", "flow"]:
        missing_idx = rng.choice(n, size=int(0.02 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 注入异常：随机位置放大偏移
    for col in ["CO", "NH3N", "flow"]:
        outlier_idx = rng.choice(n, size=int(0.01 * n), replace=False)
        df.loc[outlier_idx, col] *= rng.uniform(1.5, 2.5, size=outlier_idx.shape[0])

    return df


def main():
    ensure_directories()
    n_days = 3 * 365
    df = generate_series(n_days)
    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"示例数据已生成：{RAW_DATA_FILE}")


if __name__ == "__main__":
    main()


