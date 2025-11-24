"""
Stack 模型：使用 LightGBM 叶子索引的 One-Hot 编码作为输入，训练二级 MLP。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from models.mlp_model import MLPRegressor
from utils import load_yaml_config

logger = logging.getLogger(__name__)


class LeafStackModel:
    """叶子编码 + MLP 的二级模型。"""

    def __init__(self, config_path: str):
        cfg = load_yaml_config(config_path)
        self.config = cfg.get("stack", cfg)
        self.alpha = self.config.get("alpha", 0.5)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.feature_names: Optional[list[str]] = None
        self.mlp = MLPRegressor({"model": {k: v for k, v in self.config.items() if k != "alpha"}})

    def _to_frame(self, encoded: np.ndarray, index: pd.Index) -> pd.DataFrame:
        if self.feature_names is None or len(self.feature_names) != encoded.shape[1]:
            self.feature_names = [f"leaf_{i}" for i in range(encoded.shape[1])]
        return pd.DataFrame(encoded, index=index, columns=self.feature_names)

    def fit(
        self,
        train_leaf: np.ndarray,
        train_residual: pd.Series,
        valid_leaf: Optional[np.ndarray] = None,
        valid_residual: Optional[pd.Series] = None,
    ):
        logger.info("训练 Stack 模型，样本: %d，特征维度: %d", train_leaf.shape[0], train_leaf.shape[1])
        train_encoded = self.encoder.fit_transform(train_leaf)
        train_df = self._to_frame(train_encoded, train_residual.index)
        valid_df = None
        if (
            valid_leaf is not None
            and valid_residual is not None
            and len(valid_leaf) > 0
            and len(valid_residual) > 0
        ):
            valid_encoded = self.encoder.transform(valid_leaf)
            valid_df = self._to_frame(valid_encoded, valid_residual.index)
        self.mlp.fit(train_df, train_residual, valid_df, valid_residual)

    def predict_residual(self, leaf: np.ndarray, index: pd.Index) -> pd.Series:
        encoded = self.encoder.transform(leaf)
        feat_df = self._to_frame(encoded, index)
        return self.mlp.predict(feat_df)

    def fuse(self, lgb_pred: pd.Series, residual_pred: pd.Series) -> pd.Series:
        return lgb_pred + residual_pred * self.alpha

    def save(self, output_dir: str, model_name: str):
        os.makedirs(output_dir, exist_ok=True)
        enc_path = os.path.join(output_dir, f"{model_name}_stack_encoder.json")
        with open(enc_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "categories": [cat.tolist() for cat in self.encoder.categories_],
                    "feature_names": self.feature_names,
                    "alpha": self.alpha,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )
        self.mlp.save(output_dir, f"{model_name}_stack")
        logger.info("Stack 模型保存完成: %s", output_dir)

    def load(self, output_dir: str, model_name: str):
        enc_path = os.path.join(output_dir, f"{model_name}_stack_encoder.json")
        if not os.path.exists(enc_path):
            raise FileNotFoundError(enc_path)
        with open(enc_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        self.feature_names = meta.get("feature_names")
        self.alpha = meta.get("alpha", self.alpha)
        self.encoder.categories_ = [np.array(cat) for cat in meta["categories"]]
        self.encoder.n_features_in_ = len(self.encoder.categories_)
        self.encoder.feature_names_in_ = np.array([f"tree_{i}" for i in range(self.encoder.n_features_in_)])
        self.encoder.drop_idx_ = None
        self.encoder.sparse_output_ = False
        self.mlp.load(output_dir, f"{model_name}_stack")

