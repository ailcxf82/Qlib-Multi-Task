"""
基于 Pandas 的轻量数据集封装，以满足 qlib LGBModel 的接口要求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import Dataset


@dataclass
class Segment:
    """时间切片定义。"""

    start: str
    end: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.start, self.end)


class PandasDataset(Dataset):
    """
    通过直接传入特征与标签构建的简单数据集，实现 qlib 模型所需接口。
    """

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        segments: Dict[str, Tuple[str, str]],
    ):
        self.features = features.copy()
        self.labels = labels.copy()
        self.segments = segments
        super().__init__()

    def setup_data(self, **kwargs):
        # 无需额外初始化
        return super().setup_data(**kwargs)

    def _slice(self, segment: Union[str, Tuple[str, str]]) -> Tuple[pd.DataFrame, pd.Series]:
        if isinstance(segment, str):
            if segment not in self.segments:
                raise KeyError(f"未定义数据切片: {segment}")
            start, end = self.segments[segment]
        else:
            start, end = segment
        mask = (self.features.index.get_level_values("datetime") >= start) & (
            self.features.index.get_level_values("datetime") <= end
        )
        feat = self.features.loc[mask]
        lbl = self.labels.loc[mask]
        return feat, lbl

    def prepare(
        self,
        segments: Union[List[str], Tuple[str, ...], str, Tuple[str, str]],
        col_set="feature",
        data_key: str = DataHandlerLP.DK_I,
        **kwargs,
    ):
        if isinstance(segments, (list, tuple)) and segments and isinstance(segments[0], str):
            return [self.prepare(seg, col_set=col_set, data_key=data_key, **kwargs) for seg in segments]

        feat, lbl = self._slice(segments)
        if col_set == "feature" or (isinstance(col_set, str) and col_set.lower() == "feature"):
            return feat
        if col_set == "label":
            return lbl

        # 兼容 ["feature", "label"] 或 DataHandler.CS_ALL
        feature_block = feat
        label_block = lbl.to_frame("label")
        merged = pd.concat({"feature": feature_block, "label": label_block}, axis=1)
        return merged


