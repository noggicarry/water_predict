# -*- coding: utf-8 -*-
"""
FastAPI 推理服务：
- /health: 健康检查
- /predict: 提供最近窗口或显式窗口进行短期预测（默认使用训练配置 LOOKBACK_STEPS/FORECAST_STEPS）

启动示例（Windows / PowerShell）：
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from typing import List, Optional, Dict

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

from pathlib import Path
import sys

# 确保可从项目根导入 src 包
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    LOOKBACK_STEPS,
    FORECAST_STEPS,
    MODEL_FILE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    POOLING,
)
from src.model import LSTMForecaster
from src.data_preprocessing import load_and_resample, clean_and_impute
from src.data_preprocessing import transform_with_scaler
from src.utils import load_pickle
from src.config import SCALER_FILE


class PredictRequest(BaseModel):
    # 可选：外部提供最近窗口（原始尺度，列顺序需与 FEATURE_COLUMNS 一致）
    window: Optional[List[List[float]]] = None
    steps: Optional[int] = None  # 预测步数，默认 FORECAST_STEPS


app = FastAPI(title="水质预测 API", version="1.0.0")


@app.on_event("startup")
def load_assets() -> None:
    """加载模型与标准化器到全局。"""
    global MODEL, DEVICE, SCALER
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先准备维度（通过数据列定义）
    num_features = len(FEATURE_COLUMNS)
    num_targets = len(TARGET_COLUMNS)
    MODEL = LSTMForecaster(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pooling=POOLING,
    ).to(DEVICE)
    state = torch.load(MODEL_FILE, map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.eval()
    SCALER = load_pickle(SCALER_FILE)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """进行短期预测：
    - 若提供 window，则使用该窗口（原始尺度）
    - 否则从最新清洗数据中截取 LOOKBACK_STEPS 作为窗口
    返回：预测的 TARGET_COLUMNS 列表（原始尺度）
    """
    steps = int(req.steps) if req.steps is not None else FORECAST_STEPS

    # 准备标准化空间的种子窗口 (T, F)
    if req.window is not None:
        arr = np.array(req.window, dtype=float)
        if arr.shape != (LOOKBACK_STEPS, len(FEATURE_COLUMNS)):
            return {"error": f"window 形状应为 ({LOOKBACK_STEPS}, {len(FEATURE_COLUMNS)})"}
        # 标准化
        # 将列表转数据帧以便使用列名
        import pandas as pd
        df_last = pd.DataFrame(arr, columns=FEATURE_COLUMNS)
        df_scaled = transform_with_scaler(df_last, SCALER)
        seed_window = df_scaled.values
    else:
        # 使用系统最新清洗数据
        import pandas as pd
        df_raw = load_and_resample()
        df_clean = clean_and_impute(df_raw)
        df_scaled_all = transform_with_scaler(df_clean, SCALER)
        if len(df_scaled_all) < LOOKBACK_STEPS:
            return {"error": "历史数据不足以构造窗口"}
        seed_window = df_scaled_all.values[-LOOKBACK_STEPS:, :]

    # 迭代预测（在标准化空间），将预测写回目标维度
    target_indices = [FEATURE_COLUMNS.index(c) for c in TARGET_COLUMNS]
    preds = []
    x = seed_window.copy()
    with torch.no_grad():
        for _ in range(steps):
            xb = torch.tensor(x[None, ...], dtype=torch.float32, device=DEVICE)
            yb = MODEL(xb)
            y_np = yb.cpu().numpy()[0]
            preds.append(y_np)
            last = x[-1].copy()
            next_feat = last.copy()
            for i, idx in enumerate(target_indices):
                next_feat[idx] = y_np[i]
            x = np.vstack([x[1:], next_feat])

    # 反标准化回原空间，仅对目标列
    means = SCALER.mean_
    scales = SCALER.scale_
    preds = np.array(preds)
    restored = []
    for i, col in enumerate(TARGET_COLUMNS):
        idx = FEATURE_COLUMNS.index(col)
        restored.append(preds[:, i] * scales[idx] + means[idx])
    restored = np.stack(restored, axis=1).tolist()
    return {"targets": TARGET_COLUMNS, "predictions": restored}

# -*- coding: utf-8 -*-
"""
FastAPI 接口：提供模型预测、触发训练等功能的 API 服务。
"""
from __future__ import annotations
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd

# 将项目根目录加入模块搜索路径
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DIR, TARGET_COLUMNS
from src.predict import main as run_prediction
from src.train import main as run_training
from scripts.generate_mock_data import main as run_data_generation

# --- FastAPI 应用定义 ---
app = FastAPI(
    title="水质预测API",
    description="一个基于LSTM模型的水质指标预测服务。",
    version="1.0.0",
)

# --- Pydantic 模型定义 (用于请求体) ---
class PredictionResponse(BaseModel):
    short_term: dict
    long_term: dict

# --- 后台任务函数 ---
def background_train():
    """在后台执行训练任务。"""
    print("后台训练任务已启动...")
    run_training()
    print("后台训练任务已完成。")

def background_generate_data():
    """在后台生成模拟数据。"""
    print("后台数据生成任务已启动...")
    run_data_generation()
    print("后台数据生成任务已完成。")

# --- API Endpoints ---
@app.get("/", tags=["通用"])
def read_root():
    """欢迎接口"""
    return {"message": "欢迎使用水质预测API"}

@app.post("/predict/", response_model=PredictionResponse, tags=["预测"])
def predict():
    """
    执行短期和长期预测，并返回结果。
    
    在调用此接口前，请确保模型已经训练完毕。
    """
    try:
        print("开始执行预测...")
        run_prediction()
        
        short_fp = PROCESSED_DIR / "forecast_short.csv"
        trend_fp = PROCESSED_DIR / "forecast_trend.csv"

        if not short_fp.exists() or not trend_fp.exists():
            raise FileNotFoundError("预测结果文件未生成。")

        df_short = pd.read_csv(short_fp, index_col=0)
        df_trend = pd.read_csv(trend_fp, index_col=0)

        return {
            "short_term": df_short[TARGET_COLUMNS].to_dict(orient='index'),
            "long_term": df_trend[TARGET_COLUMNS].to_dict(orient='index')
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"依赖文件未找到: {e}。请先运行训练。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程中发生错误: {e}")

@app.post("/train/", tags=["模型管理"])
async def train_model(background_tasks: BackgroundTasks):
    """
    在后台异步触发模型训练。
    """
    background_tasks.add_task(background_train)
    return {"message": "模型训练任务已在后台启动。"}

@app.post("/generate-data/", tags=["数据管理"])
async def generate_data(background_tasks: BackgroundTasks):
    """
    在后台异步触发模拟数据生成。
    """
    background_tasks.add_task(background_generate_data)
    return {"message": "模拟数据生成任务已在后台启动。"}