"""
简单回测脚本：读取预测结果 -> 构建组合 -> 按标签收益计算净值。
"""

import argparse
import logging
import os
from typing import Optional

import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from portfolio.portfolio_builder import PortfolioBuilder
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="基于预测结果的简易回测")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="pipeline 配置文件路径",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default=None,
        help="预测结果 CSV 路径（需包含 final 列），默认自动查找最新预测文件",
    )
    parser.add_argument(
        "--industry",
        type=str,
        default=None,
        help="行业映射 CSV，包含 instrument, industry 列",
    )
    return parser.parse_args()


def load_predictions(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index(["datetime", "instrument"], inplace=True)
    return df["final"]


def load_industry(path: Optional[str]) -> Optional[pd.Series]:
    if path is None:
        return None
    df = pd.read_csv(path)
    if "instrument" not in df.columns or "industry" not in df.columns:
        raise ValueError("行业文件需包含 instrument 与 industry 列")
    return df.set_index("instrument")["industry"]


def _find_latest_prediction(prediction_dir: str) -> str:
    """自动查找最新的预测文件。"""
    import glob
    pattern = os.path.join(prediction_dir, "pred_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"在 {prediction_dir} 中未找到预测文件，请先运行 run_predict.py 或显式指定 --prediction")
    # 按修改时间排序，返回最新的
    latest = max(files, key=os.path.getmtime)
    logging.info("自动选择最新预测文件: %s", latest)
    return latest


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_yaml_config(args.config)
    
    # 自动查找最新预测文件（如果未指定）
    prediction_path = args.prediction
    if prediction_path is None:
        prediction_dir = cfg["paths"]["prediction_dir"]
        prediction_path = _find_latest_prediction(prediction_dir)
    
    preds = load_predictions(prediction_path)
    instruments = preds.index.get_level_values("instrument").unique()

    pipeline = QlibFeaturePipeline(cfg["data_config"])
    pipeline.build()
    _, labels = pipeline.get_all()
    labels = labels.loc[labels.index.isin(preds.index)]
    logging.info("labels: %s", labels)
    
    industry_map = load_industry(args.industry)
    portfolio_cfg = cfg.get("portfolio", {})
    builder = PortfolioBuilder(
        max_position=portfolio_cfg.get("max_position", 0.3),
        max_stock_weight=portfolio_cfg.get("max_stock_weight", 0.05),
        max_industry_weight=portfolio_cfg.get("max_industry_weight", 0.2),
    )

    results = []
    for dt in sorted(preds.index.get_level_values("datetime").unique()):
        score = preds.xs(dt)
        try:
            label_slice = labels.xs(dt)
        except KeyError:
            continue
        industry_slice = None
        if industry_map is not None:
            industry_slice = industry_map.reindex(score.index)
        weights = builder.build(score, industry_slice, top_k=portfolio_cfg.get("top_k", 50))
        realized = label_slice.reindex(weights.index).dropna()
        if realized.empty:
            continue
        ret = (weights.loc[realized.index] * realized).sum()
        results.append({"date": dt, "return": ret})

    if not results:
        logging.warning("未生成任何回测记录")
        return

    df = pd.DataFrame(results)
    df.sort_values("date", inplace=True)
    df["cum_return"] = (1 + df["return"]).cumprod() - 1
    stats = {
        "total_return": df["cum_return"].iloc[-1],
        "avg_return": df["return"].mean(),
        "volatility": df["return"].std(),
        "sharpe": df["return"].mean() / (df["return"].std() + 1e-8) * (252 ** 0.5),
    }

    os.makedirs(cfg["paths"]["backtest_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["backtest_dir"], "backtest_result.csv")
    df.to_csv(out_path, index=False)
    logging.info("回测完成，结果写入 %s", out_path)
    logging.info("统计指标: %s", stats)


if __name__ == "__main__":
    main()

