# -*- coding: utf-8 -*-
"""
预测脚本：
- 加载模型与标准化器
- 对测试集评估，计算 MAPE/MAE
- 进行短期与长期迭代预测，生成预警
"""
from __future__ import annotations

from typing import List, Dict
import json
from pathlib import Path

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
    ARTIFACTS_DIR,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    POOLING,
    MODEL_TYPE,
    TR_D_MODEL,
    TR_NHEAD,
    TR_NUM_LAYERS,
    TR_DIM_FF,
    TR_DROPOUT,
    TR_MAX_LEN,
)
from .utils import load_pickle, mape, mae
from .model import LSTMForecaster, TransformerForecaster
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
    if MODEL_TYPE.lower() == "transformer":
        model = TransformerForecaster(
            num_features=num_features,
            num_targets=num_targets,
            d_model=TR_D_MODEL,
            nhead=TR_NHEAD,
            num_layers=TR_NUM_LAYERS,
            dim_feedforward=TR_DIM_FF,
            dropout=TR_DROPOUT,
            pooling=POOLING,
            max_len=TR_MAX_LEN,
        ).to(device)
    else:
        model = LSTMForecaster(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            pooling=POOLING,
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
    # 逐目标指标
    per_target_metrics: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(TARGET_COLUMNS):
        y_true_col = y_test_inv[:, i]
        y_pred_col = pred_test_inv[:, i]
        # 按列计算指标（避免零除）
        col_mape = float(np.mean(np.abs((y_true_col - y_pred_col) / (np.abs(y_true_col) + 1e-6))) * 100.0)
        col_mae = float(np.mean(np.abs(y_true_col - y_pred_col)))
        per_target_metrics[col] = {"MAPE": col_mape, "MAE": col_mae}
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

    # 将评估指标与预警信息保存到 artifacts 目录
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    warnings_path = ARTIFACTS_DIR / "warnings.json"
    metrics_obj = {
        "overall": {"MAPE": float(mape_score), "MAE": float(mae_score)},
        "per_target": per_target_metrics,
        "num_test_samples": int(y_test_inv.shape[0]),
        "targets": TARGET_COLUMNS,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump(warnings, f, ensure_ascii=False, indent=2)
    print(f"指标已保存：{metrics_path}")
    print(f"预警已保存：{warnings_path}")


if __name__ == "__main__":
    main()


