# -*- coding: utf-8 -*-
"""
一键运行脚本：生成数据(若缺失) → 训练 → 预测 → 可视化。

用法（Windows / PowerShell）：
    python run_all.py

可选参数：
    --regen      强制重新生成示例数据
    --no-show    仅保存图像，不弹出窗口
    --font       指定中文字体文件路径（如 C:/Windows/Fonts/msyh.ttc）
    --verbose    输出详细进度日志
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import seaborn as sns

from src.config import (
    ensure_directories,
    RAW_DATA_FILE,
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    TARGET_COLUMNS,
    MODEL_TYPE,
    INCLUDE_TIME_FEATURES,
)


def maybe_generate_data(force: bool = False, verbose: bool = False) -> None:
    """若原始数据不存在或强制标志为真，则生成示例数据。"""
    if force or (not RAW_DATA_FILE.exists()):
        if verbose:
            print("[run_all] 生成示例数据...", flush=True)
        from scripts.generate_mock_data import main as gen_main
        gen_main()
    else:
        if verbose:
        print(f"[run_all] 检测到原始数据已存在：{RAW_DATA_FILE}", flush=True)
        print(f"[run_all] 当前模型: {MODEL_TYPE} | 时间特征: {INCLUDE_TIME_FEATURES}", flush=True)


def train_model(verbose: bool = False) -> None:
    """调用现有训练入口。"""
    if verbose:
        print("[run_all] 开始训练...", flush=True)
    from src.train import main as train_main
    train_main()
    if verbose:
        print("[run_all] 训练完成。", flush=True)


def predict_and_save(verbose: bool = False) -> None:
    """调用现有预测入口，保存短期/趋势预测结果与预警信息。"""
    if verbose:
        print("[run_all] 开始预测...", flush=True)
    from src.predict import main as predict_main
    predict_main()
    if verbose:
        print("[run_all] 预测完成。", flush=True)


def setup_chinese_font(font_path: str | None = None) -> None:
    """在 Windows 优先尝试字体文件注册，其次按字体名称匹配。"""
    try:
        # 如果用户显式提供了字体文件路径，优先使用
        if font_path:
            p = Path(font_path)
            if p.exists():
                font_manager.fontManager.addfont(str(p))
                name = font_manager.FontProperties(fname=str(p)).get_name()
                rcParams["font.sans-serif"] = [name]
                rcParams["font.family"] = [name]
                rcParams["axes.unicode_minus"] = False
                try:
                    font_manager._rebuild()
                except Exception:
                    pass
                print(f"已启用用户字体文件: {p} -> {name}")
                return
            else:
                print(f"提供的字体文件不存在：{p}")

        file_candidates = [
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\msyh.ttf",
            r"C:\\Windows\\Fonts\\msyhbd.ttc",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simsun.ttc",
            r"C:\\Windows\\Fonts\\Deng.ttf",
        ]
        for fp in file_candidates:
            p = Path(fp)
            if p.exists():
                font_manager.fontManager.addfont(str(p))
                name = font_manager.FontProperties(fname=str(p)).get_name()
                rcParams["font.sans-serif"] = [name]
                rcParams["font.family"] = [name]
                rcParams["axes.unicode_minus"] = False
                try:
                    font_manager._rebuild()
                except Exception:
                    pass
                print(f"已启用中文字体文件: {p} -> {name}")
                return

        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                rcParams["font.sans-serif"] = [name]
                rcParams["font.family"] = [name]
                rcParams["axes.unicode_minus"] = False
                print(f"已设置中文字体: {name}")
                return
        print("未检测到可用中文字体，可能出现方框。建议在系统安装 '微软雅黑' 或 '思源黑体'。")
    except Exception as e:
        print(f"设置中文字体时出现警告：{e}")


def visualize(save_only: bool = False, font_path: str | None = None, verbose: bool = False) -> Path:
    """读取清洗与预测结果，绘制并保存多图页。返回保存路径。"""
    # 仅保存图像时，切换无界面后端，避免 Tkinter 相关告警
    if save_only:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass
    clean_fp = PROCESSED_DIR / "clean.csv"
    short_fp = PROCESSED_DIR / "forecast_short.csv"
    trend_fp = PROCESSED_DIR / "forecast_trend.csv"

    if not clean_fp.exists():
        raise FileNotFoundError("未找到清洗数据 clean.csv，请先完成训练/预测。")
    if not short_fp.exists() or not trend_fp.exists():
        raise FileNotFoundError("未找到预测结果，请先完成预测。")

    df_clean = pd.read_csv(clean_fp, parse_dates=[0], index_col=0)
    df_short = pd.read_csv(short_fp, parse_dates=[0], index_col=0)
    df_trend = pd.read_csv(trend_fp, parse_dates=[0], index_col=0)

    # 仅展示最后 180 天历史，便于观察趋势
    history_window = 180
    sns.set(style="whitegrid")
    # seaborn.set 可能重置字体，这里再次应用中文字体设置
    setup_chinese_font(font_path)
    n_rows = len(TARGET_COLUMNS)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, 4 * n_rows),
        sharex=False,
        constrained_layout=True,  # 自动压缩子图间距
    )
    if n_rows == 1:
        axes = [axes]

    for i, col in enumerate(TARGET_COLUMNS):
        ax = axes[i]
        hist = df_clean[col].dropna().iloc[-history_window:]
        hist.plot(ax=ax, label=f"历史 {col}")
        if col in df_short.columns:
            df_short[col].plot(ax=ax, label=f"短期预测 {col}")
        if col in df_trend.columns:
            df_trend[col].plot(ax=ax, label=f"长期趋势 {col}")
        ax.set_title(f"{col} 历史与预测")
        ax.legend(loc="best")
        ax.set_xlabel("时间")
        ax.set_ylabel(col)
        # 去除轴域左右空白边
        ax.margins(x=0)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "overview.png"
    # 保存时裁剪外部空白
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    if verbose:
        print(f"[run_all] 可视化图像已保存：{out_path}", flush=True)
    if not save_only:
        plt.show()
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="水质预测一键运行")
    parser.add_argument("--regen", action="store_true", help="强制重新生成示例数据")
    parser.add_argument("--no-show", action="store_true", help="仅保存图像，不弹窗显示")
    parser.add_argument("--font", type=str, default=r"C:\\Windows\\Fonts\\msyh.ttc", help="指定中文字体文件路径，例如 C:/Windows/Fonts/msyh.ttc")
    parser.add_argument("--verbose", action="store_true", help="输出详细进度日志")
    args = parser.parse_args()

    if args.verbose:
        print("[run_all] 启动一键流程...", flush=True)
    ensure_directories()
    maybe_generate_data(force=args.regen, verbose=args.verbose)
    train_model(verbose=args.verbose)
    predict_and_save(verbose=args.verbose)
    visualize(save_only=args.no_show, font_path=args.font, verbose=args.verbose)
    if args.verbose:
        print("[run_all] 全部完成。", flush=True)


if __name__ == "__main__":
    main()


