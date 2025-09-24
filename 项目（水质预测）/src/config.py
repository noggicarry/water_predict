# -*- coding: utf-8 -*-
"""
配置模块：集中管理路径、特征名单、训练与预测超参数。
"""
from pathlib import Path


# 基础路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# 文件路径
RAW_DATA_FILE = RAW_DIR / "water_quality.csv"
SCALER_FILE = ARTIFACTS_DIR / "standard_scaler.pkl"
MODEL_FILE = MODELS_DIR / "lstm_water_quality.pth"

# 列配置
TIME_COL = "timestamp"
FEATURE_COLUMNS = [
    "CO",    # 一氧化碳（示例指标，可按实际替换或补充）
    "NH3N",  # 氨氮
    "pH",
    "DO",    # 溶解氧
    "flow"   # 流量
]

# 预测目标（可以只选择部分关键指标）
TARGET_COLUMNS = ["CO", "NH3N", "flow"]

# 频率与窗口
FREQ = "D"              # 示例按天聚合：'H' 表示小时
LOOKBACK_STEPS = 30      # 回看窗口长度（天）
FORECAST_STEPS = 14      # 默认短期预测步数（天）
TREND_STEPS = 90         # 长期趋势预测步数（天）

# 数据切分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# 训练超参数
RANDOM_SEED = 42
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
POOLING = "last"  # 可选: "last" | "mean" | "attn"
CLIP_GRAD_NORM = 1.0  # 梯度裁剪阈值；<=0 表示不裁剪
LR_SCHEDULER_FACTOR = 0.5  # ReduceLROnPlateau 衰减因子
LR_SCHEDULER_PATIENCE = 2   # 学习率调度耐心
LR_MIN = 1e-6               # 最小学习率

# 模型选择与优化
MODEL_TYPE = "lstm"  # 可选: "lstm" | "transformer"
WEIGHT_DECAY = 1e-4   # 优化器权重衰减

# Transformer 相关超参数（当 MODEL_TYPE="transformer" 生效）
TR_D_MODEL = 128
TR_NHEAD = 4
TR_NUM_LAYERS = 2
TR_DIM_FF = 256
TR_DROPOUT = 0.2
TR_MAX_LEN = 1000

# 时间特征工程
INCLUDE_TIME_FEATURES = False  # 是否添加时间正余弦特征

# 预警阈值（可按实际标准调整）
THRESHOLDS = {
    "CO": 2.0,      # 示例阈值
    "NH3N": 1.0,    # 示例阈值
    "flow": 1.2     # 示例阈值（可理解为上限倍数/绝对值）
}


def ensure_directories() -> None:
    """确保所需目录存在。"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


