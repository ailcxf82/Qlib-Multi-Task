"""
运行预测流程：加载最新模型 -> 计算多模型预测 -> 输出融合结果。
"""

import argparse
import logging
import os
from typing import Dict

import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from predictor.predictor import PredictorEngine
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib 因子模型预测")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml")
    # parser.add_argument("--start", type=str, required=True, help="预测起始日期")
    # parser.add_argument("--end", type=str, required=True, help="预测结束日期")
    # parser.add_argument("--tag", type=str, default="auto", help="模型标识，auto 则使用最新窗口")
    
    parser.add_argument(
        "--start",
        type=str,
        default=os.environ.get("RUN_PRED_START", "2018-01-01"),
        help="预测起始日期，默认 2018-01-01，可通过环境变量 RUN_PRED_START 覆盖",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=os.environ.get("RUN_PRED_END", "2018-01-31"),
        help="预测结束日期，默认 2018-01-31，可通过环境变量 RUN_PRED_END 覆盖",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=os.environ.get("RUN_PRED_TAG", "auto"),
        help="模型标识，默认 auto，可通过环境变量 RUN_PRED_TAG 覆盖",
    )
    return parser.parse_args()


def _load_ic_histories(log_path: str) -> Dict[str, pd.Series]:
    if not os.path.exists(log_path):
        today = pd.Timestamp.today()
        base = pd.Series([0.1], index=[today])
        return {"lgb": base, "mlp": base, "stack": base}
    df = pd.read_csv(log_path, parse_dates=["valid_end"])
    histories = {
        "lgb": pd.Series(df["ic_lgb"].values, index=df["valid_end"]),
        "mlp": pd.Series(df["ic_mlp"].values, index=df["valid_end"]),
        "stack": pd.Series(df["ic_stack"].values, index=df["valid_end"]),
    }
    return histories


def _latest_tag(log_path: str) -> str:
    if not os.path.exists(log_path):
        raise FileNotFoundError("无法自动推断模型标识，请显式传入 --tag")
    df = pd.read_csv(log_path)
    latest = df.iloc[-1]["valid_end"].replace("-", "")
    return latest


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_yaml_config(args.config)
    log_path = os.path.join(cfg["paths"]["log_dir"], "training_metrics.csv")
    tag = _latest_tag(log_path) if args.tag == "auto" else args.tag
    ic_histories = _load_ic_histories(log_path)

    pipeline = QlibFeaturePipeline(cfg["data_config"])
    pipeline.build()
    features, _ = pipeline.get_slice(args.start, args.end)
    predictor = PredictorEngine(args.config)
    predictor.load_models(tag)
    final_pred, preds, weights = predictor.predict(features, ic_histories)
    predictor.save_predictions(final_pred, preds, f"{tag}_{args.start}_{args.end}")
    logging.info("IC 动态权重: %s", weights)


if __name__ == "__main__":
    main()


