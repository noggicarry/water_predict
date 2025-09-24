# -*- coding: utf-8 -*-
"""
滚动回测评估（Walk-Forward Backtest）：
 - 基于全量清洗后的数据，使用固定标准化器（训练集拟合）与已训练模型
 - 以滑动窗口作为起点，进行 FORECAST_STEPS 迭代预测
 - 计算整体与分步(1..H)的 MAPE/MAE，以及分目标指标

运行示例（Windows / PowerShell）：
    python -m src.backtest --step 7 --max-origins 100
输出：
    artifacts/backtest.json  （主指标）
    artifacts/backtest_per_step.tsv （每步汇总）
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import argparse
import json
import numpy as np
import pandas as pd
import torch

from .config import (
    ensure_directories,
    PROCESSED_DIR,
    SCALER_FILE,
    MODEL_FILE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    LOOKBACK_STEPS,
    FORECAST_STEPS,
    FREQ,
    ARTIFACTS_DIR,
    # 模型/超参
    MODEL_TYPE,
    POOLING,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    TR_D_MODEL,
    TR_NHEAD,
    TR_NUM_LAYERS,
    TR_DIM_FF,
    TR_DROPOUT,
    TR_MAX_LEN,
    INCLUDE_TIME_FEATURES,
)
from .data_preprocessing import (
    load_and_resample,
    clean_and_impute,
    transform_with_scaler,
)
from .data_preprocessing import add_time_features  # 时间特征（若启用）
from .utils import load_pickle, mape, mae
from .model import LSTMForecaster, TransformerForecaster


def inverse_targets(z_pred: np.ndarray, scaler, target_cols: List[str], all_cols: List[str]) -> np.ndarray:
    """仅对目标列进行标准化逆变换（基于已有 StandardScaler 参数）。"""
    col_index_map = {c: i for i, c in enumerate(all_cols)}
    means = scaler.mean_
    scales = scaler.scale_
    restored = []
    for i, col in enumerate(target_cols):
        idx = col_index_map[col]
        restored.append(z_pred[:, i] * scales[idx] + means[idx])
    return np.stack(restored, axis=1)


def build_model(num_features: int, num_targets: int, device: torch.device):
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
    return model


def walk_forward_backtest(step: int = 7, max_origins: int | None = None) -> Dict:
    ensure_directories()
    # 数据与标准化
    df_raw = load_and_resample()
    df_clean = clean_and_impute(df_raw)
    if INCLUDE_TIME_FEATURES:
        df_clean, _added = add_time_features(df_clean, FREQ)

    scaler = load_pickle(SCALER_FILE)
    df_scaled = transform_with_scaler(df_clean, scaler)
    all_cols = list(df_scaled.columns)
    target_indices = [all_cols.index(c) for c in TARGET_COLUMNS]

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_features=len(all_cols), num_targets=len(TARGET_COLUMNS), device=device)
    state = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state)
    model.eval()

    T = len(df_scaled)
    horizon = FORECAST_STEPS
    origins = []
    t = LOOKBACK_STEPS
    while t + horizon <= T:
        origins.append(t)
        t += max(1, step)
        if max_origins is not None and len(origins) >= max_origins:
            break

    if not origins:
        raise RuntimeError("数据长度不足，无法进行滚动回测。")

    per_origin_overall: List[Dict[str, float]] = []
    per_step_records: List[Dict[str, float]] = []  # 聚合时使用
    per_target_step: Dict[str, List[List[float]]] = {c: [] for c in TARGET_COLUMNS}

    with torch.no_grad():
        for o in origins:
            seed = df_scaled.values[o - LOOKBACK_STEPS:o, :]
            # 迭代预测（标准化空间）
            preds = []
            x = seed.copy()
            for _ in range(horizon):
                xb = torch.tensor(x[None, ...], dtype=torch.float32, device=device)
                yb = model(xb)
                y_np = yb.cpu().numpy()[0]
                preds.append(y_np)
                last = x[-1].copy()
                next_feat = last.copy()
                for i, idx in enumerate(target_indices):
                    next_feat[idx] = y_np[i]
                x = np.vstack([x[1:], next_feat])
            preds = np.array(preds)  # (H, num_targets)

            # 反标准化与取真实值
            preds_inv = inverse_targets(preds, scaler, TARGET_COLUMNS, all_cols)
            y_true = df_clean[TARGET_COLUMNS].values[o:o + horizon, :]

            # 整体指标（该起点、全步平均）
            origin_mape = mape(y_true, preds_inv)
            origin_mae = mae(y_true, preds_inv)
            per_origin_overall.append({"MAPE": float(origin_mape), "MAE": float(origin_mae)})

            # 每步指标（跨目标平均）
            for h in range(horizon):
                step_mape = mape(y_true[h:h+1, :], preds_inv[h:h+1, :])
                step_mae = mae(y_true[h:h+1, :], preds_inv[h:h+1, :])
                per_step_records.append({"step": h + 1, "MAPE": float(step_mape), "MAE": float(step_mae)})

            # 分目标-分步
            for j, col in enumerate(TARGET_COLUMNS):
                series = []
                for h in range(horizon):
                    y_t = y_true[h, j]
                    y_p = preds_inv[h, j]
                    mape_jh = float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + 1e-6))) * 100.0)
                    series.append(mape_jh)
                per_target_step[col].append(series)

    # 聚合
    overall_mape = float(np.mean([d["MAPE"] for d in per_origin_overall]))
    overall_mae = float(np.mean([d["MAE"] for d in per_origin_overall]))

    # 每步聚合（跨所有起点平均）
    df_steps = pd.DataFrame(per_step_records)
    per_step = (
        df_steps.groupby("step")["MAPE", "MAE"].mean().reset_index().to_dict(orient="records")
    )

    # 分目标-分步 MAPE 平均
    per_target = {}
    for col, mat in per_target_step.items():
        arr = np.array(mat)  # (num_origins, H)
        per_target[col] = [float(x) for x in arr.mean(axis=0)]

    # 保存
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS_DIR / "backtest.json"
    out_tsv = ARTIFACTS_DIR / "backtest_per_step.tsv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": MODEL_TYPE,
                "freq": FREQ,
                "lookback": LOOKBACK_STEPS,
                "horizon": horizon,
                "num_origins": len(origins),
                "overall": {"MAPE": overall_mape, "MAE": overall_mae},
                "per_step": per_step,
                "per_target_mape": per_target,
                "targets": TARGET_COLUMNS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # DataFrame 保存每步指标
    df_steps.groupby("step")["MAPE", "MAE"].mean().to_csv(out_tsv, sep="\t")
    print(f"回测完成：origins={len(origins)} overall_MAPE={overall_mape:.2f}% overall_MAE={overall_mae:.4f}")
    print(f"结果保存：{out_json}\n每步指标：{out_tsv}")

    return {
        "overall_MAPE": overall_mape,
        "overall_MAE": overall_mae,
        "num_origins": len(origins),
    }


def main():
    parser = argparse.ArgumentParser(description="滚动回测评估")
    parser.add_argument("--step", type=int, default=7, help="起点步进（单位=时间步）")
    parser.add_argument("--max-origins", type=int, default=None, help="最多评估的起点数")
    args = parser.parse_args()
    walk_forward_backtest(step=args.step, max_origins=args.max_origins)


if __name__ == "__main__":
    main()


