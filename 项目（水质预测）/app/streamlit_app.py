# -*- coding: utf-8 -*-
"""
Streamlit 可视化：对清洗数据、预测结果（短期/趋势）进行可视化与预警摘要展示。
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# 将项目根目录加入模块搜索路径，避免从子目录运行时找不到 src 包
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DIR, TARGET_COLUMNS, FEATURE_COLUMNS


st.set_page_config(page_title="水质预测与调控", layout="wide")
st.title("水质预测与调控（LSTM）")

clean_fp = PROCESSED_DIR / "clean.csv"
short_fp = PROCESSED_DIR / "forecast_short.csv"
trend_fp = PROCESSED_DIR / "forecast_trend.csv"

cols = st.columns(2)
with cols[0]:
    st.subheader("数据与特征")
    if clean_fp.exists():
        df_clean = pd.read_csv(clean_fp, parse_dates=[0], index_col=0)
        st.dataframe(df_clean.tail(10))
        select_cols = st.multiselect("选择查看的列", FEATURE_COLUMNS, default=TARGET_COLUMNS)
        st.line_chart(df_clean[select_cols])
    else:
        st.info("尚未生成清洗数据，请先运行训练或预测脚本。")

with cols[1]:
    st.subheader("预测结果：短期与趋势")
    if short_fp.exists():
        df_short = pd.read_csv(short_fp, parse_dates=[0], index_col=0)
        st.markdown("**短期预测（默认 14 天）**")
        st.line_chart(df_short[TARGET_COLUMNS])
    else:
        st.info("暂无短期预测结果。")
    if trend_fp.exists():
        df_trend = pd.read_csv(trend_fp, parse_dates=[0], index_col=0)
        st.markdown("**长期趋势（默认 90 天）**")
        st.line_chart(df_trend[TARGET_COLUMNS])
    else:
        st.info("暂无长期趋势结果。")

st.caption("提示：如首次使用，请运行 scripts/generate_mock_data.py 与 src/train.py、src/predict.py 生成必要文件。")


