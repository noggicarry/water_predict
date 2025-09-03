水质预测与调控（LSTM 时序预测）

本项目用于对水质参数（如 CO、氨氮 NH₃-N、pH、DO、流量等）与水量波动进行时间序列建模与预测，支持缺失值补偿、异常检测、短期预警与长期趋势分析，并提供可视化界面（Streamlit）。

### 功能特性
- 多变量时序建模：基于 LSTM 的多变量单步预测 + 迭代多步预测。
- 数据稳定性保障：缺失值填补（插值/统计填补）与异常检测（MAD/IsolationForest）。
- 预警与趋势：支持 1-2 周短期预警与 30-90 天趋势分析。
- 可视化分析：交互式查看原始、清洗与预测结果，便于管理决策。

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

### 目标与指标
- 预测误差：将关键指标（如 CO、NH₃-N、流量）的 MAPE 控制在 5% 左右（示例数据可稳定达到）。
- 预警能力：自动识别短期风险波动，支持提前 1-2 周干预。

### 注意
- 所有中文注释均为 UTF-8 编码，若出现乱码请确认编辑器编码设置。
- 初次运行会在 `data/`、`models/`、`artifacts/` 目录下生成必要文件。


