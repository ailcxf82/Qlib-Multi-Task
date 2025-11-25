"""
预测模块：负责加载已训练模型、计算单模型预测并执行 IC 动态加权。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import pandas as pd

from models.ensemble_manager import EnsembleModelManager
from models.stack_model import LeafStackModel
from predictor.weight_dynamic import RankICDynamicWeighter
from utils import load_yaml_config

logger = logging.getLogger(__name__)


class PredictorEngine:
    """预测管线封装。"""

    def __init__(self, pipeline_config: str):
        cfg = load_yaml_config(pipeline_config)
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.ensemble = EnsembleModelManager(cfg, cfg.get("ensemble"))
        self.stack = LeafStackModel(cfg["stack_config"])
        ic_cfg = cfg.get("ic_logging", {})
        self.weighter = RankICDynamicWeighter(
            window=ic_cfg.get("window", 60),
            half_life=ic_cfg.get("half_life", 20),
            min_weight=ic_cfg.get("min_weight", 0.05),
            max_weight=ic_cfg.get("max_weight", 0.7),
            clip_negative=ic_cfg.get("clip_negative", True),
        )

    def load_models(self, tag: str):
        model_dir = self.paths["model_dir"]
        logger.info("加载模型，标识: %s", tag)
        self.ensemble.load(model_dir, tag)
        self.stack.load(model_dir, tag)

    def predict(
        self,
        features: pd.DataFrame,
        ic_histories: Dict[str, pd.Series],
    ) -> Tuple[pd.Series, Dict[str, pd.Series], Dict[str, float]]:
        """返回融合预测、各模型预测以及权重。"""
        blend_pred, base_preds, aux = self.ensemble.predict(features)
        lgb_pred = base_preds.get("lgb")
        lgb_leaf = aux.get("lgb")
        if lgb_pred is None or lgb_leaf is None:
            raise RuntimeError("需要 LightGBM 预测以驱动 Stack 模型，请检查 ensemble 配置")
        residual_pred = self.stack.predict_residual(lgb_leaf, features.index)
        stack_pred = self.stack.fuse(lgb_pred, residual_pred)
        preds = dict(base_preds)
        preds["stack"] = stack_pred
        if blend_pred is not None:
            preds["qlib_ensemble"] = blend_pred
        # 根据历史 IC 计算动态权重，兼顾稳定性
        weights = self.weighter.get_weights(ic_histories)
        final_pred = self.weighter.blend(preds, weights)
        return final_pred, preds, weights

    def save_predictions(self, final_pred: pd.Series, preds: Dict[str, pd.Series], tag: str):
        os.makedirs(self.paths["prediction_dir"], exist_ok=True)
        out_path = os.path.join(self.paths["prediction_dir"], f"pred_{tag}.csv")
        df = pd.DataFrame({"final": final_pred})
        for name, series in preds.items():
            df[name] = series
        # 统一索引顺序为 (datetime, instrument)，便于后续回测与查看
        if df.index.nlevels == 2:
            try:
                df = df.reorder_levels(["datetime", "instrument"]).sort_index()
            except KeyError:
                # 如果 MultiIndex 未命名，则按照位置强制调换
                df = df.reorder_levels([1, 0]).sort_index()
        # MultiIndex 直接写入 csv，便于后续回测按日期/证券读取
        df.to_csv(out_path, index_label=["datetime", "instrument"])
        logger.info("预测结果已保存: %s", out_path)

