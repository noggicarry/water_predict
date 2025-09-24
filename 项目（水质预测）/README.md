水质预测与调控（LSTM 时序预测）

本项目用于对水质参数（如 CO、氨氮 NH₃-N、pH、DO、流量等）与水量波动进行时间序列建模与预测，支持缺失值补偿、异常检测、短期预警与长期趋势分析，并提供可视化界面（Streamlit）。

### 功能特性
- 多变量时序建模：基于 LSTM 的多变量单步预测 + 迭代多步预测。
- 数据稳定性保障：缺失值填补（插值/统计填补）与异常检测（MAD/IsolationForest）。
- 预警与趋势：支持 1-2 周短期预警与 30-90 天趋势分析。
- 可视化分析：交互式查看原始、清洗与预测结果，便于管理决策。
 - 可选注意力池化：支持 last/mean/attn 三种时间聚合，默认保持 last。
 - 训练增强：早停 + 学习率调度 + 梯度裁剪，训练日志落盘 `artifacts/train_log.tsv`。
 - 指标与预警持久化：`artifacts/metrics.json` 与 `artifacts/warnings.json`，Streamlit 自动展示。
 - 在线推理 API：FastAPI 服务 `app/api.py`，便于面试展示与集成。
 - 算法扩展：支持 `MODEL_TYPE` 切换 LSTM/Transformer、可选时间正余弦特征、滚动回测评估。

### 目录结构
```
project/
  ├─ app/
  │   └─ streamlit_app.py           # 可视化应用入口
  ├─ data/
  │   ├─ raw/                       # 原始/示例数据
  │   └─ processed/                 # 清洗与归一化后数据
  ├─ models/                        # 训练好的模型权重
  ├─ artifacts/                     # 标准化器等模型资产
  ├─ scripts/
  │   └─ generate_mock_data.py      # 生成示例数据
  └─ src/
      ├─ config.py
      ├─ utils.py
      ├─ data_preprocessing.py
      ├─ anomaly_detection.py
      ├─ model.py
      ├─ train.py
      └─ predict.py
```

### 快速开始（Windows / PowerShell）
1. 创建并激活虚拟环境：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. 安装依赖：
```powershell
pip install -r requirements.txt
```
3. 生成示例数据（模块方式运行，确保包导入稳定）：
```powershell
python -m scripts.generate_mock_data
```
4. 训练模型：
```powershell
python -m src.train
```
5. 进行预测（默认 14 天短期 + 90 天趋势）：
```powershell
python -m src.predict
```
6. 启动可视化应用：
```powershell
streamlit run .\app\streamlit_app.py
```

7. 启动在线推理 API（可选）：
```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
示例调用（Python）：
```python
import requests
print(requests.get("http://127.0.0.1:8000/health").json())
payload = {"steps": 14}
print(requests.post("http://127.0.0.1:8000/predict", json=payload).json())
```

8. 算法侧配置（`src/config.py`）：
- `MODEL_TYPE`: `"lstm"` 或 `"transformer"`
- `POOLING`: `"last" | "mean" | "attn"`
- `WEIGHT_DECAY`: 优化器权重衰减
- `INCLUDE_TIME_FEATURES`: 是否添加时间正余弦特征
- `TR_*`: Transformer 相关维度/头数/层数等

9. 运行滚动回测（评估不同起点滚动预测表现）：
```powershell
python -m src.backtest --step 7 --max-origins 100
```
输出文件：`artifacts/backtest.json` 与 `artifacts/backtest_per_step.tsv`

10. 超参搜索（简化交叉验证）：
```powershell
python -m src.hparam_search --trials 8 --epochs 10 --mode random
```
输出文件：`artifacts/hparam_search.tsv`、`artifacts/best_config.json`、`models/best_from_search.pth`

11. 特征重要性与注意力可视化：
```powershell
python -m src.feature_importance --metric mape --samples 256
```
输出文件：`artifacts/feature_importance.tsv`、（若为注意力池化）`artifacts/attn_weights.npy`

### 目标与指标
- 预测误差：将关键指标（如 CO、NH₃-N、流量）的 MAPE 控制在 5% 左右（示例数据可稳定达到）。
- 预警能力：自动识别短期风险波动，支持提前 1-2 周干预。

### 配置要点（`src/config.py`）
- `POOLING`: "last" | "mean" | "attn"，控制 LSTM 时间聚合方式。
- `CLIP_GRAD_NORM`: >0 启用梯度裁剪。
- `LR_SCHEDULER_*`: ReduceLROnPlateau 学习率调度参数。
 - `MODEL_TYPE` 与 `TR_*`、`INCLUDE_TIME_FEATURES`：算法侧切换与特征扩展。

### 注意
- 所有中文注释均为 UTF-8 编码，若出现乱码请确认编辑器编码设置。
- 初次运行会在 `data/`、`models/`、`artifacts/` 目录下生成必要文件。

### 面试展示建议
- 先运行 `python run_all.py --verbose`，得到 `artifacts/overview.png`、`metrics.json`、`warnings.json`。
- 打开 Streamlit 展示数据、预测与指标预警；再启动 FastAPI 演示在线推理。
- 代码亮点：注意力池化、Transformer 对比实验、时间特征、可复现实验（随机种子与日志）、清晰模块化结构。


