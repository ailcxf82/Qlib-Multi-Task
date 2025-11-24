"""
IC 动态加权模块：对各模型历史 rank-IC 进行加权计算，生成投资时的组合权重。
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RankICDynamicWeighter:
    """根据 rank-IC 历史表现分配权重。"""

    def __init__(
        self,
        window: int = 60,
        half_life: int = 20,
        min_weight: float = 0.05,
        max_weight: float = 0.7,
        clip_negative: bool = True,
    ):
        self.window = window
        self.half_life = half_life
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.clip_negative = clip_negative

    @staticmethod
    def compute_rank_ic(pred: pd.Series, label: pd.Series) -> float:
        """计算单期 rank-IC。"""
        if pred.empty or label.empty:
            return np.nan
        aligned = pred.align(label, join="inner")
        if aligned[0].empty:
            return np.nan
        return aligned[0].rank().corr(aligned[1], method="spearman")

    def _ic_ir(self, ic_series: pd.Series) -> float:
        """根据半衰期计算 IC-IR。"""
        ic_series = ic_series.dropna().tail(self.window)
        if len(ic_series) < 2:
            return 0.0
        # 按半衰期生成指数衰减权重，越近的 IC 权重越大
        weights = np.array([0.5 ** (i / max(1, self.half_life - 1)) for i in range(len(ic_series))])[::-1]
        weights /= weights.sum()
        mean = np.sum(ic_series.values * weights)
        std = np.sqrt(np.sum(weights * (ic_series.values - mean) ** 2))
        if std == 0:
            return 0.0
        return mean / std

    def get_weights(self, ic_histories: Dict[str, pd.Series]) -> Dict[str, float]:
        """将 IC 序列映射为最终权重。"""
        scores = {name: self._ic_ir(series) for name, series in ic_histories.items()}
        if self.clip_negative:
            scores = {k: max(0.0, v) for k, v in scores.items()}
        total = sum(scores.values())
        if total == 0:
            # 回退为等权
            eq = 1.0 / max(1, len(scores))
            return {k: eq for k in scores}
        weights = {k: v / total for k, v in scores.items()}
        # 施加 min/max 约束
        weights = {k: np.clip(w, self.min_weight, self.max_weight) for k, w in weights.items()}
        total = sum(weights.values())
        return {k: w / total for k, w in weights.items()}

    def blend(self, preds: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
        """依据权重融合预测结果。"""
        combined = None  # 逐个累加，保持索引对齐
        for name, series in preds.items():
            w = weights.get(name, 0.0)
            contrib = series * w
            combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)
        return combined


