# -*- coding: utf-8 -*-
"""
特征重要性与注意力可视化：
- 方案 A（通用）：遮挡法（occlusion）评估特征对验证集 MAE/MAPE 的影响
- 方案 B（注意力）：当使用 pooling=attn 时，导出时间维权重分布的可视化数据

运行示例：
    python -m src.feature_importance --metric mape --samples 256
输出：
    artifacts/feature_importance.tsv
    artifacts/attn_weights.npy （若模型为 attn 池化）
"""
from __future__ import annotations

from typing import Dict, List

import argparse
import numpy as np
import pandas as pd
import torch

from .config import (
    ensure_directories,
    ARTIFACTS_DIR,
    MODEL_FILE,
    POOLING,
)
from .data_preprocessing import prepare_datasets
from .model import LSTMForecaster, TransformerForecaster
from .utils import mape, mae


def select_model_example() -> torch.nn.Module:
    # 利用 prepare_datasets 的 meta 推断维度，并从磁盘加载当前已训练模型
    (splits, meta, _map) = prepare_datasets()
    num_features = meta["num_features"]
    num_targets = meta["num_targets"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 简化：尝试两类模型的构建（权重键不匹配时将抛错，这里优先 LSTM）
    try:
        model = LSTMForecaster(num_features=num_features, num_targets=num_targets, pooling=POOLING).to(device)
        state = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        model = TransformerForecaster(num_features=num_features, num_targets=num_targets, pooling=POOLING).to(device)
        state = torch.load(MODEL_FILE, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model


def occlusion_importance(metric: str = "mape", max_samples: int | None = 256) -> pd.DataFrame:
    (splits, meta, col_map) = prepare_datasets()
    X_val, y_val = splits["val"]
    all_columns = meta.get("all_columns")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = select_model_example()

    # 子采样验证集，避免运行过慢
    if max_samples is not None and len(X_val) > max_samples:
        X_val = X_val[:max_samples]
        y_val = y_val[:max_samples]

    Xb = torch.tensor(X_val, dtype=torch.float32, device=device)
    yb = torch.tensor(y_val, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_base = model(Xb).cpu().numpy()
    y_true = yb.cpu().numpy()

    def compute_metric(a, b):
        return mape(a, b) if metric.lower() == "mape" else mae(a, b)

    base_score = compute_metric(y_true, pred_base)

    rows = []
    # 遮挡每个特征：将该特征置为 0（标准化空间零均值附近），衡量性能劣化
    for col, idx in col_map.items():
        X_mask = Xb.clone()
        X_mask[:, :, idx] = 0.0
        with torch.no_grad():
            pred_mask = model(X_mask).cpu().numpy()
        score = compute_metric(y_true, pred_mask)
        rows.append({"feature": col, "score": float(score), "delta": float(score - base_score)})

    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "feature_importance.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"特征重要性已保存：{out}")
    return df


def export_attention_weights(max_samples: int | None = 128):
    # 仅当使用 attn 池化时有意义
    if POOLING.lower() != "attn":
        print("当前非注意力池化，跳过注意力权重导出。")
        return None
    (splits, meta, _map) = prepare_datasets()
    X_val, _ = splits["val"]
    if max_samples is not None and len(X_val) > max_samples:
        X_val = X_val[:max_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = select_model_example()

    # 尝试访问 attn_score，若不存在则跳过
    if not hasattr(model, "attn_score"):
        print("模型不包含 attn_score 层，跳过导出。")
        return None

    Xb = torch.tensor(X_val, dtype=torch.float32, device=device)
    with torch.no_grad():
        # 前向到 LSTM/Transformer 编码输出
        # 这里复用模型 forward 不易拿到中间层，示范性地对 LSTM 情况重新计算编码输出
        if hasattr(model, "lstm"):
            out, _ = model.lstm(Xb)
        elif hasattr(model, "encoder") and hasattr(model, "input_proj") and hasattr(model, "pos_encoding"):
            h = model.input_proj(Xb)
            h = model.pos_encoding(h)
            out = model.encoder(h)
        else:
            print("暂不支持该模型的注意力导出。")
            return None
        scores = model.attn_score(out)
        weights = torch.softmax(scores, dim=1).cpu().numpy()
    npy_path = ARTIFACTS_DIR / "attn_weights.npy"
    np.save(npy_path, weights)
    print(f"注意力权重已导出：{npy_path}")
    return npy_path


def main():
    parser = argparse.ArgumentParser(description="特征重要性与注意力可视化")
    parser.add_argument("--metric", type=str, default="mape", choices=["mape", "mae"], help="评价指标")
    parser.add_argument("--samples", type=int, default=256, help="用于评估/导出的最大样本数")
    args = parser.parse_args()
    ensure_directories()
    occlusion_importance(metric=args.metric, max_samples=args.samples)
    export_attention_weights(max_samples=min(args.samples, 128))


if __name__ == "__main__":
    main()


