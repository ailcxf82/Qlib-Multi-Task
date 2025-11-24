"""
LightGBM 模型封装，调用 qlib.contrib.model.gbdt.LGBModel 训练并输出叶子索引。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel

from utils import load_yaml_config
from utils.dataset import PandasDataset
logger = logging.getLogger(__name__)


class LightGBMModelWrapper:
    """结合 qlib LGBModel 的二次封装。"""

    def __init__(self, config_path: str):
        cfg = load_yaml_config(config_path)
        self.config = cfg["model"]
        self.model = LGBModel(
            loss=self.config.get("loss", "mse"),
            num_boost_round=self.config.get("num_boost_round", 1000),
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            **self.config.get("params", {}),
        )
        self.booster: Optional[lgb.Booster] = None
        self.feature_names: Optional[list[str]] = None

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        # 通过 PandasDataset 向 qlib 声明训练/验证时间切片
        segments = {
            "train": (
                train_feat.index.get_level_values("datetime").min(),
                train_feat.index.get_level_values("datetime").max(),
            )
        }
        features = train_feat
        labels = train_label
        has_valid = (
            valid_feat is not None
            and valid_label is not None
            and len(valid_feat) > 0
            and len(valid_label) > 0
        )
        if has_valid:
            segments["valid"] = (
                valid_feat.index.get_level_values("datetime").min(),
                valid_feat.index.get_level_values("datetime").max(),
            )
            features = pd.concat([train_feat, valid_feat], axis=0)
            labels = pd.concat([train_label, valid_label], axis=0)
        else:
            logger.warning("验证集为空，LightGBM 将仅使用训练数据")

        dataset = PandasDataset(features=features, labels=labels, segments=segments)
        logger.info("开始训练 LightGBM，训练样本: %d", len(train_feat))
        self.model.fit(dataset=dataset)
        self.booster = self.model.model
        self.feature_names = list(train_feat.columns)

    def predict(self, feat: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        if self.booster is None:
            raise RuntimeError("模型尚未训练")
        values = feat.values
        preds = self.booster.predict(values)
        # pred_leaf=True 返回每棵树的叶子编号，用作二级模型输入
        leaf_index = self.booster.predict(values, pred_leaf=True)
        return pd.Series(preds, index=feat.index, name="lgb_pred"), leaf_index

    def save(self, output_dir: str, model_name: str):
        if self.booster is None:
            raise RuntimeError("无可保存模型")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_lgb.txt")
        meta_path = os.path.join(output_dir, f"{model_name}_lgb_meta.json")
        self.booster.save_model(model_path)
        meta = {
            "config": self.config,
            "feature_names": self.feature_names,
        }
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        logger.info("LightGBM 模型已保存: %s", model_path)

    def load(self, output_dir: str, model_name: str):
        model_path = os.path.join(output_dir, f"{model_name}_lgb.txt")
        meta_path = os.path.join(output_dir, f"{model_name}_lgb_meta.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.booster = lgb.Booster(model_file=model_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            self.feature_names = meta.get("feature_names")
        logger.info("LightGBM 模型已加载: %s", model_path)

