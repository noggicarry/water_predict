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