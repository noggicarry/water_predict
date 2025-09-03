# -*- coding: utf-8 -*-
"""
预测脚本：
- 加载模型与标准化器
- 对测试集评估，计算 MAPE/MAE
- 进行短期与长期迭代预测，生成预警
"""
from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from .config import (
    ensure_directories,
    PROCESSED_DIR,
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    LOOKBACK_STEPS,
    FORECAST_STEPS,
    TREND_STEPS,
    THRESHOLDS,
    FREQ,
)
from .utils import load_pickle, mape, mae
from .model import LSTMForecaster
from .data_preprocessing import prepare_datasets, transform_with_scaler, load_and_resample, clean_and_impute


def inverse_targets(z_pred: np.ndarray, scaler, target_cols: List[str], feature_cols: List[str]) -> np.ndarray:
    """仅对目标列进行标准化逆变换（基于 StandardScaler 参数）。"""
    col_index_map = {c: i for i, c in enumerate(feature_cols)}
    means = scaler.mean_
    scales = scaler.scale_
    restored = []
    for i, col in enumerate(target_cols):
        idx = col_index_map[col]
        restored.append(z_pred[:, i] * scales[idx] + means[idx])
    return np.stack(restored, axis=1)


def iterative_forecast(
    model: LSTMForecaster,
    seed_window: np.ndarray,
    steps: int,
    target_indices: list,
) -> np.ndarray:
    """基于最后窗口进行迭代预测（标准化空间），将预测写回目标维度。"""
    device = next(model.parameters()).device
    model.eval()
    preds = []
    x = seed_window.copy()  # (T, F)
    with torch.no_grad():
        for _ in range(steps):
            xb = torch.tensor(x[None, ...], dtype=torch.float32, device=device)
            yb = model(xb)  # (1, num_targets)
            y_np = yb.cpu().numpy()[0]
            preds.append(y_np)
            last = x[-1].copy()
            next_feat = last.copy()
            for i, idx in enumerate(target_indices):
                next_feat[idx] = y_np[i]
            x = np.vstack([x[1:], next_feat])
    return np.array(preds)


def main():
    ensure_directories()

    # 预处理并获取数据与元信息
    (splits, meta, col_index_map) = prepare_datasets()
    num_features = meta["num_features"]
    num_targets = meta["num_targets"]

    # 组网并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(
        num_features=num_features,
        num_targets=num_targets,
    ).to(device)
    state = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state)

    # 评估测试集单步性能
    X_test, y_test = splits["test"]
    with torch.no_grad():
        pred_test = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    scaler = load_pickle(SCALER_FILE)
    y_test_inv = inverse_targets(y_test, scaler, TARGET_COLUMNS, FEATURE_COLUMNS)
    pred_test_inv = inverse_targets(pred_test, scaler, TARGET_COLUMNS, FEATURE_COLUMNS)
    mape_score = mape(y_test_inv, pred_test_inv)
    mae_score = mae(y_test_inv, pred_test_inv)
    print(f"测试集: MAPE={mape_score:.2f}% MAE={mae_score:.4f}")

    # 构造迭代预测的种子窗口（使用全体清洗+标准化后的末尾窗口）
    df_raw = load_and_resample()
    df_clean = clean_and_impute(df_raw)
    scaler = load_pickle(SCALER_FILE)
    df_scaled_all = transform_with_scaler(df_clean, scaler)
    seed_window = df_scaled_all.values[-LOOKBACK_STEPS:, :]

    # 进行短期与长期预测（在标准化空间内）
    target_indices = [col_index_map[c] for c in TARGET_COLUMNS]

    # 实现迭代预测
    def roll_forecast(seed: np.ndarray, steps: int) -> np.ndarray:
        preds = []
        x = seed.copy()
        with torch.no_grad():
            for _ in range(steps):
                xb = torch.tensor(x[None, ...], dtype=torch.float32, device=device)
                yb = model(xb)
                y_np = yb.cpu().numpy()[0]  # (num_targets,)
                preds.append(y_np)
                last = x[-1].copy()
                # 将目标维度替换为预测值，形成下一时刻的特征向量
                next_feat = last.copy()
                for i, idx in enumerate(target_indices):
                    next_feat[idx] = y_np[i]
                # 更新窗口
                x = np.vstack([x[1:], next_feat])
        return np.array(preds)

    preds_short_z = roll_forecast(seed_window, FORECAST_STEPS)
    preds_trend_z = roll_forecast(seed_window, TREND_STEPS)

    # 反标准化回原空间
    preds_short = inverse_targets(preds_short_z, scaler, TARGET_COLUMNS, FEATURE_COLUMNS)
    preds_trend = inverse_targets(preds_trend_z, scaler, TARGET_COLUMNS, FEATURE_COLUMNS)

    # 保存预测结果
    from pandas.tseries.frequencies import to_offset
    idx_short = pd.date_range(df_clean.index[-1] + to_offset(FREQ), periods=FORECAST_STEPS, freq=FREQ)
    idx_trend = pd.date_range(df_clean.index[-1] + to_offset(FREQ), periods=TREND_STEPS, freq=FREQ)
    df_short = pd.DataFrame(preds_short, index=idx_short, columns=TARGET_COLUMNS)
    df_trend = pd.DataFrame(preds_trend, index=idx_trend, columns=TARGET_COLUMNS)
    df_short.to_csv(PROCESSED_DIR / "forecast_short.csv")
    df_trend.to_csv(PROCESSED_DIR / "forecast_trend.csv")

    # 生成简易预警：超阈值标记
    warnings: Dict[str, List[str]] = {}
    for name, df_pred in {"short": df_short, "trend": df_trend}.items():
        notes = []
        for col in TARGET_COLUMNS:
            thr = THRESHOLDS.get(col)
            if thr is None:
                continue
            exceed = df_pred[df_pred[col] > thr]
            if not exceed.empty:
                first_day = exceed.index[0].strftime("%Y-%m-%d")
                notes.append(f"{name} 预测 {col} 首次超阈值时间：{first_day}")
        warnings[name] = notes

    for k, v in warnings.items():
        if v:
            print(f"[{k} 预警]")
            for line in v:
                print("- ", line)
        else:
            print(f"[{k} 预警] 无超阈值风险。")


if __name__ == "__main__":
    main()


