"""
投资组合构建模块，应用仓位、单股、行业约束。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioBuilder:
    """通过简单加权 + 约束裁剪生成组合。"""

    def __init__(
        self,
        max_position: float = 0.3,
        max_stock_weight: float = 0.05,
        max_industry_weight: float = 0.2,
    ):
        self.max_position = max_position
        self.max_stock_weight = max_stock_weight
        self.max_industry_weight = max_industry_weight

    def build(
        self,
        scores: pd.Series,
        industry_map: Optional[pd.Series] = None,
        top_k: int = 100,
    ) -> pd.Series:
        """根据预测得分生成权重。"""
        # 1. 选取 top_k，使用线性衰减权重
        filtered = scores.dropna().sort_values(ascending=False).head(top_k)
        ranks = filtered.rank(ascending=False, method="first")
        weights = (top_k - ranks + 1) / top_k
        # 2. 控制整体仓位
        weights = weights / weights.sum() * self.max_position
        # 3. 单股权重裁剪
        weights = weights.clip(upper=self.max_stock_weight)
        weights = weights / weights.sum() * self.max_position

        if industry_map is not None:
            weights = self._apply_industry_constraint(weights, industry_map)

        logger.info("组合共选股 %d，实际仓位 %.2f%%", len(weights), weights.sum() * 100)
        return weights

    def _apply_industry_constraint(self, weights: pd.Series, industry_map: pd.Series) -> pd.Series:
        industries = industry_map.reindex(weights.index)
        constrained = weights.copy()
        for industry, group in industries.groupby(industries):
            idx = group.index
            total = constrained.loc[idx].sum()
            limit = self.max_industry_weight
            if total > limit:
                scale = limit / total
                constrained.loc[idx] *= scale
        # 行业约束后需要再次归一
        constrained = constrained / constrained.sum() * self.max_position
        return constrained


