"""
滚动训练器：串联特征、模型，输出多模型权重与指标。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from models.ensemble_manager import EnsembleModelManager
from models.stack_model import LeafStackModel
from utils import load_yaml_config

logger = logging.getLogger(__name__)


@dataclass
class Window:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


def _rank_ic(pred: pd.Series, label: pd.Series) -> float:
    pred, label = pred.align(label, join="inner")
    if pred.empty:
        return float("nan")
    return pred.rank().corr(label, method="spearman")


class RollingTrainer:
    """核心训练流程。"""

    def __init__(self, pipeline_config: str):
        self.cfg = load_yaml_config(pipeline_config)
        self.paths = self.cfg["paths"]
        self.data_cfg_path = self.cfg["data_config"]
        self.pipeline = QlibFeaturePipeline(self.data_cfg_path)
        self.ensemble = EnsembleModelManager(self.cfg, self.cfg.get("ensemble"))
        self.stack = LeafStackModel(self.cfg["stack_config"])

    def _generate_windows(self) -> Iterable[Window]:
        rolling = self.cfg["rolling"]
        data_cfg = load_yaml_config(self.data_cfg_path)["data"]
        start = pd.Timestamp(data_cfg["start_time"])
        end = pd.Timestamp(data_cfg["end_time"])
        train_offset = pd.DateOffset(months=rolling["train_months"])
        valid_offset = pd.DateOffset(months=rolling["valid_months"])
        step = pd.DateOffset(months=rolling["step_months"])

        # cursor 指向验证起点，前推 train_offset 即训练区间
        cursor = start + train_offset
        while cursor + valid_offset <= end:
            train_start = cursor - train_offset
            train_end = cursor - pd.Timedelta(days=1)
            valid_start = cursor
            valid_end = cursor + valid_offset - pd.Timedelta(days=1)
            yield Window(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=valid_end.strftime("%Y-%m-%d"),
            )
            cursor += step

    def _slice(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        start: str,
        end: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        idx = features.index
        mask = (idx.get_level_values("datetime") >= start) & (idx.get_level_values("datetime") <= end)
        feat = features.loc[mask]
        lbl = labels.loc[mask]
        return feat, lbl

    def train(self):
        self.pipeline.build()
        features, labels = self.pipeline.get_all()
        os.makedirs(self.paths["model_dir"], exist_ok=True)
        os.makedirs(self.paths["log_dir"], exist_ok=True)
        metrics: List[Dict] = []

        for idx, window in enumerate(self._generate_windows()):
            logger.info("==== 滚动窗口 %d: %s -> %s ====", idx, window.train_start, window.valid_end)
            train_feat, train_lbl = self._slice(features, labels, window.train_start, window.train_end)
            valid_feat, valid_lbl = self._slice(features, labels, window.valid_start, window.valid_end)
            if len(train_feat) < self.cfg["rolling"].get("min_samples", 1000):
                logger.warning("训练样本不足，跳过该窗口")
                continue
            has_valid = valid_feat is not None and not valid_feat.empty and valid_lbl is not None and not valid_lbl.empty
            if not has_valid:
                logger.warning("窗口 %d 验证集为空或不足，退化为仅训练", idx)
                valid_feat = None
                valid_lbl = None

            # 统一训练多模型
            self.ensemble.fit(train_feat, train_lbl, valid_feat, valid_lbl)

            _, train_preds, train_aux = self.ensemble.predict(train_feat)
            lgb_train_pred = train_preds.get("lgb")
            lgb_train_leaf = train_aux.get("lgb")
            if lgb_train_pred is None or lgb_train_leaf is None:
                raise RuntimeError("LeafStackModel 需要 LightGBM 输出，请在 ensemble.models 中包含 `lgb`")
            valid_blend = valid_preds = valid_aux = None
            if has_valid:
                valid_blend, valid_preds, valid_aux = self.ensemble.predict(valid_feat)

            valid_pred = valid_leaf = None
            if valid_preds is not None:
                valid_pred = valid_preds.get("lgb")
            if valid_aux is not None:
                valid_leaf = valid_aux.get("lgb")

            # residual = label - lgb，用于二级学习
            train_leaf = lgb_train_leaf
            train_residual = train_lbl - lgb_train_pred
            valid_residual = None if (not has_valid or valid_pred is None) else valid_lbl - valid_pred
            self.stack.fit(train_leaf, train_residual, valid_leaf, valid_residual)

            # 计算验证集 IC
            if has_valid:
                mlp_valid_pred = valid_preds.get("mlp") if valid_preds is not None else None
                stack_residual = self.stack.predict_residual(valid_leaf, valid_feat.index) if valid_leaf is not None else None
                stack_valid_pred = (
                    self.stack.fuse(valid_pred, stack_residual)
                    if (valid_pred is not None and stack_residual is not None)
                    else None
                )
                metric = {
                    "window": idx,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "valid_start": window.valid_start,
                    "valid_end": window.valid_end,
                    "ic_lgb": _rank_ic(valid_pred, valid_lbl) if valid_pred is not None else float("nan"),
                    "ic_mlp": _rank_ic(mlp_valid_pred, valid_lbl) if mlp_valid_pred is not None else float("nan"),
                    "ic_stack": _rank_ic(stack_valid_pred, valid_lbl) if stack_valid_pred is not None else float("nan"),
                    "ic_qlib_ensemble": _rank_ic(valid_blend, valid_lbl) if valid_blend is not None else float("nan"),
                }
                metrics.append(metric)

            # 以验证区间结束日作为模型文件名，方便按日期加载
            tag = window.valid_end.replace("-", "")
            self.ensemble.save(self.paths["model_dir"], tag)
            self.stack.save(self.paths["model_dir"], tag)

        if metrics:
            # 记录所有窗口的 IC，用于后续动态加权或评估
            df = pd.DataFrame(metrics)
            df.to_csv(os.path.join(self.paths["log_dir"], "training_metrics.csv"), index=False)
            logger.info("训练指标已保存，共 %d 条记录", len(df))
        else:
            logger.warning("未产出任何训练窗口指标")

