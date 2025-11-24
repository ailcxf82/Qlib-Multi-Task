"""
基于 qlib 的特征提取流水线，负责：
1. 初始化 qlib 环境
2. 调用 D.features 获取行情与因子
3. 生成对齐标签 Ref($close, -5)/$close - 1
4. 进行基础标准化，并输出训练用 DataFrame
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

from utils import load_yaml_config

logger = logging.getLogger(__name__)


class QlibFeaturePipeline:
    """特征管线核心类。"""

    def __init__(self, config_path: str):
        self.config = load_yaml_config(config_path)
        self._init_qlib()
        self.feature_cfg = self.config["data"]
        self.features_df: pd.DataFrame | None = None
        self.label_series: pd.Series | None = None
        self._feature_mean: pd.Series | None = None
        self._feature_std: pd.Series | None = None

    def _init_qlib(self):
        qlib_cfg = self.config.get("qlib", {})
        if qlib.is_initialized():
            # 在 notebook/调试环境中可能重复调用，避免重复初始化
            return
        logger.info("初始化 qlib，数据目录: %s", qlib_cfg.get("provider_uri"))
        qlib.init(
            provider_uri=qlib_cfg.get("provider_uri"),
            region=qlib_cfg.get("region", "cn"),
            expression_cache=None,
        )

    def build(self):
        """执行特征提取。"""
        feats = self.feature_cfg["features"]
        instruments = self._parse_instruments(self.feature_cfg["instruments"])
        start = self.feature_cfg["start_time"]
        end = self.feature_cfg["end_time"]
        freq = self.feature_cfg.get("freq", "day")
        label_expr = self.feature_cfg.get("label", "Ref($close, -5)/$close - 1")

        logger.info("提取特征: %s", feats)
        feature_panel = D.features(instruments=instruments, fields=feats, start_time=start, end_time=end, freq=freq)
        label_panel = D.features(instruments=instruments, fields=[label_expr], start_time=start, end_time=end, freq=freq)

        feature_panel.columns = feats
        label_series = label_panel.iloc[:, 0].rename("label")

        # 基础对齐
        # inner join + dropna 保证特征、标签完全对齐
        combined = feature_panel.join(label_series, how="inner").dropna()
        features = combined.drop(columns=["label"])
        label = combined["label"]

        self._fit_norm(features)
        norm_feat = self._transform(features)

        self.features_df = norm_feat
        self.label_series = label
        logger.info("特征构建完成，样本量: %d", len(norm_feat))

    def _fit_norm(self, features: pd.DataFrame):
        """计算全局均值方差。"""
        self._feature_mean = features.mean()
        # 避免标准差为 0 导致除零
        std = features.std().replace(0, 1)
        self._feature_std = std

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """应用标准化。"""
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("标准化参数尚未拟合，请先调用 build()")
        arr = (features - self._feature_mean) / self._feature_std
        return arr.clip(-5, 5)  # 简单去极值，避免极端噪声

    def get_slice(self, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series]:
        """按时间切片返回特征。"""
        if self.features_df is None or self.label_series is None:
            raise RuntimeError("尚未构建特征，请先调用 build()")
        idx = self.features_df.index
        mask = (idx.get_level_values("datetime") >= start) & (idx.get_level_values("datetime") <= end)
        feat = self.features_df.loc[mask]
        lbl = self.label_series.loc[mask]
        return feat, lbl

    def get_all(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.features_df is None or self.label_series is None:
            raise RuntimeError("尚未构建特征，请先调用 build()")
        return self.features_df, self.label_series

    def stats(self) -> Dict[str, pd.Series]:
        """返回标准化统计量，供落地保存/加载。"""
        return {
            "mean": self._feature_mean,
            "std": self._feature_std,
        }

    @staticmethod
    def _parse_instruments(inst_conf: Union[str, Dict[str, Any], Tuple[str, ...], list[str]]) -> Union[Dict[str, Any], list[str]]:
        """
        将配置的股票池转换为 qlib 支持的输入。

        - 字符串默认视作市场别名（如 "csi300"），转换为 {"market": xxx, "filter_pipe": []}
        - 已是 dict 的情况下直接返回（便于自定义过滤器）
        - list/tuple 视为具体股票代码集合
        """
        if isinstance(inst_conf, str):
            return {"market": inst_conf, "filter_pipe": []}
        if isinstance(inst_conf, dict):
            return inst_conf
        if isinstance(inst_conf, (list, tuple)):
            return list(inst_conf)
        raise ValueError(f"不支持的股票池配置类型: {type(inst_conf)}")

