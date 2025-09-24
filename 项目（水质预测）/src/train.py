# -*- coding: utf-8 -*-
"""
训练脚本：
- 加载并预处理数据
- 训练 LSTM 模型并早停
- 保存最佳模型与资产
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .config import (
    ensure_directories,
    RANDOM_SEED,
    BATCH_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_FILE,
    ARTIFACTS_DIR,
    POOLING,
    CLIP_GRAD_NORM,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    LR_MIN,
    MODEL_TYPE,
    WEIGHT_DECAY,
    TR_D_MODEL,
    TR_NHEAD,
    TR_NUM_LAYERS,
    TR_DIM_FF,
    TR_DROPOUT,
    TR_MAX_LEN,
)
from .utils import seed_all
from .data_preprocessing import prepare_datasets
from .model import LSTMForecaster, TransformerForecaster


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        # 梯度裁剪（可选）
        if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def main():
    ensure_directories()
    seed_all(RANDOM_SEED)

    (splits, meta, _col_map) = prepare_datasets()
    num_features = meta["num_features"]
    num_targets = meta["num_targets"]

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    # 与 PyTorch 期望的形状一致
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

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

    criterion = nn.L1Loss()  # MAE 对异常值不敏感
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 学习率调度：验证集无改进时衰减学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_MIN,
        verbose=True,
    )

    best_val = float("inf")
    patience = EARLY_STOPPING_PATIENCE
    best_state = None

    # 训练日志（tab 分隔）：epoch, train_loss, val_loss, lr
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.6f}")
        log_lines.append(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{current_lr:.8f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = EARLY_STOPPING_PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("早停触发，停止训练。")
                break

        # 按验证损失调度学习率
        scheduler.step(val_loss)

    # 保存最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"最佳模型已保存：{MODEL_FILE}")

    # 保存训练日志
    log_file = ARTIFACTS_DIR / "train_log.tsv"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch\ttrain_loss\tval_loss\tlr\n")
        if log_lines:
            f.write("\n".join(log_lines))
    print(f"训练日志已保存：{log_file}")


if __name__ == "__main__":
    main()


