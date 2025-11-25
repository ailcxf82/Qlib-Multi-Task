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
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder

from models.mlp_model import MLPRegressor
from utils import load_yaml_config

logger = logging.getLogger(__name__)


class LeafStackModel:
    """叶子编码 + MLP 的二级模型，支持 OneHot 或哈希压缩编码。"""

    def __init__(self, config_path: str):
        cfg = load_yaml_config(config_path)
        self.config = cfg.get("stack", cfg)
        self.alpha = self.config.get("alpha", 0.5)
        self.encoding = self.config.get("encoding", "hashing").lower()
        self.hash_dim = self.config.get("hash_dim", 1024)
        self.encoder: Optional[OneHotEncoder] = None
        self.hasher: Optional[FeatureHasher] = None
        if self.encoding == "onehot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        elif self.encoding == "hashing":
            self.hasher = FeatureHasher(
                n_features=self.hash_dim,
                input_type="dict",
                dtype=np.float32,
            )
        else:
            raise ValueError(f"不支持的叶子编码方式: {self.encoding}")
        exclude_keys = {"alpha", "encoding", "hash_dim"}
        self.feature_names: Optional[list[str]] = None
        self.mlp = MLPRegressor({"model": {k: v for k, v in self.config.items() if k not in exclude_keys}})

    def _to_frame(self, encoded: np.ndarray, index: pd.Index, prefix: str = "leaf") -> pd.DataFrame:
        if self.feature_names is None or len(self.feature_names) != encoded.shape[1]:
            self.feature_names = [f"{prefix}_{i}" for i in range(encoded.shape[1])]
        return pd.DataFrame(encoded, index=index, columns=self.feature_names)

    def _hash_leaf(self, leaf: np.ndarray, index: pd.Index) -> pd.DataFrame:
        if self.hasher is None:
            raise RuntimeError("哈希编码器未初始化")
        rows = ({f"t{j}_{leaf_val}": 1 for j, leaf_val in enumerate(sample)} for sample in leaf)
        encoded_sparse = self.hasher.transform(rows)
        encoded = encoded_sparse.toarray()
        return self._to_frame(encoded, index, prefix="hash")

    def fit(
        self,
        train_leaf: np.ndarray,
        train_residual: pd.Series,
        valid_leaf: Optional[np.ndarray] = None,
        valid_residual: Optional[pd.Series] = None,
    ):
        logger.info("训练 Stack 模型，样本: %d，原始叶子维度: %d", train_leaf.shape[0], train_leaf.shape[1])
        if self.encoding == "onehot":
            if self.encoder is None:
                raise RuntimeError("OneHotEncoder 未初始化")
            train_encoded = self.encoder.fit_transform(train_leaf)
            train_df = self._to_frame(train_encoded, train_residual.index, prefix="leaf")
        else:
            train_df = self._hash_leaf(train_leaf, train_residual.index)
        valid_df = None
        if (
            valid_leaf is not None
            and valid_residual is not None
            and len(valid_leaf) > 0
            and len(valid_residual) > 0
        ):
            if self.encoding == "onehot":
                valid_encoded = self.encoder.transform(valid_leaf)
                valid_df = self._to_frame(valid_encoded, valid_residual.index, prefix="leaf")
            else:
                valid_df = self._hash_leaf(valid_leaf, valid_residual.index)
        self.mlp.fit(train_df, train_residual, valid_df, valid_residual)

    def predict_residual(self, leaf: np.ndarray, index: pd.Index) -> pd.Series:
        # 线上阶段只需 transform（不再 fit）
        if self.encoding == "onehot":
            if self.encoder is None:
                raise RuntimeError("OneHotEncoder 未初始化")
            encoded = self.encoder.transform(leaf)
            feat_df = self._to_frame(encoded, index, prefix="leaf")
        else:
            feat_df = self._hash_leaf(leaf, index)
        return self.mlp.predict(feat_df)

    def fuse(self, lgb_pred: pd.Series, residual_pred: pd.Series) -> pd.Series:
        # 通过 residual 学习补充 LGB 的结构化短板
        return lgb_pred + residual_pred * self.alpha

    def save(self, output_dir: str, model_name: str):
        os.makedirs(output_dir, exist_ok=True)
        enc_path = os.path.join(output_dir, f"{model_name}_stack_encoder.json")
        with open(enc_path, "w", encoding="utf-8") as fp:
            meta = {
                "feature_names": self.feature_names,
                "alpha": self.alpha,
                "encoding": self.encoding,
                "hash_dim": self.hash_dim,
            }
            if self.encoding == "onehot" and self.encoder is not None:
                meta["categories"] = [cat.tolist() for cat in self.encoder.categories_]
            json.dump(meta, fp, ensure_ascii=False, indent=2)
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
        self.encoding = meta.get("encoding", self.encoding)
        self.hash_dim = meta.get("hash_dim", self.hash_dim)
        if self.encoding == "onehot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            categories = meta.get("categories")
            if categories is None:
                raise ValueError("缺少 OneHot categories")
            self.encoder.categories_ = [np.array(cat) for cat in categories]
            self.encoder.n_features_in_ = len(self.encoder.categories_)
            # sklearn OneHotEncoder 在推理阶段需要补回 feature_names_in_ 等属性
            self.encoder.feature_names_in_ = np.array([f"tree_{i}" for i in range(self.encoder.n_features_in_)])
            self.encoder.drop_idx_ = None
            self.encoder.sparse_output_ = False
            self.hasher = None
        else:
            self.encoder = None
            self.hasher = FeatureHasher(
                n_features=self.hash_dim,
                input_type="dict",
                dtype=np.float32,
            )
        self.mlp.load(output_dir, f"{model_name}_stack")

