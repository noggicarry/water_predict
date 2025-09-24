# -*- coding: utf-8 -*-
"""
超参搜索（简化交叉验证版）：
- 在给定搜索空间内随机/网格抽样若干 trial
- 每个 trial 在 train/val 上训练若干 epoch（较小 EPOCHS），以验证集损失选最优
- 输出：artifacts/hparam_search.tsv 与 artifacts/best_config.json；最佳权重 models/best_from_search.pth

运行示例（Windows / PowerShell）:
    python -m src.hparam_search --trials 8 --epochs 10 --mode random
"""
from __future__ import annotations

from typing import Dict, Any, List

import argparse
import itertools
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .config import (
    ensure_directories,
    ARTIFACTS_DIR,
    MODEL_FILE,
    BATCH_SIZE,
    RANDOM_SEED,
    CLIP_GRAD_NORM,
    LR_MIN,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)
from .data_preprocessing import prepare_datasets
from .model import LSTMForecaster, TransformerForecaster
from .utils import seed_all


def build_model(cfg: Dict[str, Any], num_features: int, num_targets: int, device: torch.device):
    model_type = cfg.get("model_type", "lstm").lower()
    pooling = cfg.get("pooling", "last")
    if model_type == "transformer":
        model = TransformerForecaster(
            num_features=num_features,
            num_targets=num_targets,
            d_model=cfg.get("tr_d_model", 128),
            nhead=cfg.get("tr_nhead", 4),
            num_layers=cfg.get("tr_layers", 2),
            dim_feedforward=cfg.get("tr_dim_ff", 256),
            dropout=cfg.get("dropout", 0.2),
            pooling=pooling,
            max_len=cfg.get("tr_max_len", 1000),
        )
    else:
        model = LSTMForecaster(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pooling=pooling,
        )
    return model.to(device)


def train_eval(cfg: Dict[str, Any], epochs: int, device: torch.device) -> Dict[str, Any]:
    (splits, meta, _col_map) = prepare_datasets()
    num_features = meta["num_features"]
    num_targets = meta["num_targets"]

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(cfg, num_features, num_targets, device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_MIN,
        verbose=False,
    )

    def run_epoch(loader, train: bool) -> float:
        if train:
            model.train()
        else:
            model.eval()
        total = 0.0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if train:
                    optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                if train:
                    loss.backward()
                    if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                    optimizer.step()
                total += loss.item() * xb.size(0)
        return total / len(loader.dataset)

    best_val = float("inf")
    best_state = None
    for _ in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_loss": float(best_val), "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "cfg": cfg}


def grid(space: Dict[str, List[Any]]):
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield {k: v for k, v in zip(keys, values)}


def random_space(space: Dict[str, List[Any]], trials: int, seed: int) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    keys = list(space.keys())
    configs = []
    for _ in range(trials):
        cfg = {k: rnd.choice(space[k]) for k in keys}
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description="超参搜索")
    parser.add_argument("--trials", type=int, default=8, help="尝试次数（随机搜索）或网格大小")
    parser.add_argument("--epochs", type=int, default=10, help="每个 trial 训练的 epoch 数（较小以便快速搜索）")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "grid"], help="搜索模式")
    args = parser.parse_args()

    ensure_directories()
    seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 基本搜索空间（可按需扩展）
    space = {
        "model_type": ["lstm", "transformer"],
        "pooling": ["last", "mean", "attn"],
        "hidden_size": [64, 128, 256],
        "num_layers": [1, 2, 3],
        "tr_d_model": [64, 128],
        "tr_nhead": [2, 4],
        "tr_layers": [1, 2],
        "tr_dim_ff": [128, 256, 512],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [5e-4, 1e-3, 2e-3],
    }

    if args.mode == "grid":
        configs = list(grid(space))
    else:
        configs = random_space(space, trials=args.trials, seed=RANDOM_SEED)

    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for i, cfg in enumerate(configs, 1):
        print(f"[trial {i}/{len(configs)}] cfg={cfg}")
        res = train_eval(cfg, epochs=args.epochs, device=device)
        res_row = {"trial": i, "val_loss": res["val_loss"], **cfg}
        results.append(res_row)
        if (best is None) or (res["val_loss"] < best["val_loss"]):
            best = {"val_loss": res["val_loss"], "cfg": cfg, "state_dict": res["state_dict"]}

    # 落盘
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tsv = ARTIFACTS_DIR / "hparam_search.tsv"
    import pandas as pd
    pd.DataFrame(results).sort_values("val_loss").to_csv(tsv, sep="\t", index=False)
    print(f"搜索结果已保存：{tsv}")

    if best is not None:
        best_json = ARTIFACTS_DIR / "best_config.json"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({"val_loss": best["val_loss"], "cfg": best["cfg"]}, f, ensure_ascii=False, indent=2)
        # 保存最佳权重（不覆盖正式模型）
        out_model = MODEL_FILE.parent / "best_from_search.pth"
        torch.save(best["state_dict"], out_model)
        print(f"最佳配置：val_loss={best['val_loss']:.4f} 已保存 {best_json}，权重：{out_model}")


if __name__ == "__main__":
    main()


