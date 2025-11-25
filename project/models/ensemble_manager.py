"""
Qlib 多模型协同封装：负责统一训练/预测接口，并通过 qlib 的 Ensemble 算法完成加权融合。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from qlib.model.ens.ensemble import AverageEnsemble

from models.model_registry import create_model


class _QlibAverageAdapter:
    """封装 qlib AverageEnsemble，使其可以直接处理 pd.Series 预测结果。"""

    def __init__(self):
        self._ensemble = AverageEnsemble()

    def __call__(self, preds: Dict[str, pd.Series]) -> pd.Series:
        if not preds:
            raise ValueError("无可融合的预测结果")
        if len(preds) == 1:
            return next(iter(preds.values())).rename("qlib_ensemble")
        formatted = {name: series.to_frame(name) for name, series in preds.items()}
        result = self._ensemble(formatted)
        if isinstance(result, pd.DataFrame):
            # 默认只关心单列融合结果；若存在多列，则取平均。
            return result.mean(axis=1).rename("qlib_ensemble")
        if isinstance(result, pd.Series):
            return result.rename("qlib_ensemble")
        raise TypeError(f"无法识别的 qlib Ensemble 输出类型: {type(result)}")


class EnsembleAggregator:
    """将 qlib Ensemble 策略适配为 Series -> Series 的接口。"""

    STRATEGY_MAP = {
        "average": _QlibAverageAdapter,
    }

    def __init__(self, strategy: str = "average"):
        strategy = (strategy or "average").lower()
        adapter_cls = self.STRATEGY_MAP.get(strategy)
        if adapter_cls is None:
            raise ValueError(f"暂不支持的 Ensemble 策略: {strategy}")
        self._adapter = adapter_cls()

    def aggregate(self, preds: Dict[str, pd.Series]) -> pd.Series:
        return self._adapter(preds)


class EnsembleModelManager:
    """
    统一管理多模型训练与预测。

    config 示例:
    ensemble:
      aggregator: average
      models:
        - name: lgb
          type: lightgbm
          config_key: lightgbm_config
        - name: mlp
          type: mlp
          config_key: mlp_config
    """

    def __init__(self, pipeline_cfg: Dict, ensemble_cfg: Optional[Dict] = None):
        self.pipeline_cfg = pipeline_cfg
        self.ensemble_cfg = ensemble_cfg or {}
        self.models = OrderedDict()
        self._build_models()
        aggregator_strategy = (self.ensemble_cfg or {}).get("aggregator", "average")
        self.aggregator: Optional[EnsembleAggregator] = None
        if aggregator_strategy and aggregator_strategy != "disabled":
            self.aggregator = EnsembleAggregator(aggregator_strategy)

    def _resolve_config_path(self, spec: Dict) -> str:
        if "config" in spec:
            return spec["config"]
        config_key = spec.get("config_key")
        if config_key and config_key in self.pipeline_cfg:
            return self.pipeline_cfg[config_key]
        raise ValueError(f"模型 {spec.get('name')} 未提供 config 或 config_key")

    def _default_specs(self) -> List[Dict]:
        return [
            {"name": "lgb", "type": "lightgbm", "config_key": "lightgbm_config"},
            {"name": "mlp", "type": "mlp", "config_key": "mlp_config"},
        ]

    def _build_models(self):
        specs = (self.ensemble_cfg or {}).get("models")
        if not specs:
            specs = self._default_specs()
        for spec in specs:
            name = spec["name"]
            model_type = spec["type"]
            cfg_path = self._resolve_config_path(spec)
            self.models[name] = create_model(model_type, cfg_path)

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        for model in self.models.values():
            model.fit(train_feat, train_label, valid_feat, valid_label)

    def predict(self, feat: pd.DataFrame) -> Tuple[Optional[pd.Series], Dict[str, pd.Series], Dict[str, object]]:
        preds: Dict[str, pd.Series] = {}
        aux: Dict[str, object] = {}
        for name, model in self.models.items():
            output = model.predict(feat)
            if isinstance(output, tuple):
                preds[name], aux[name] = output
            else:
                preds[name] = output
        blended = None
        if self.aggregator is not None:
            blended = self.aggregator.aggregate(preds)
        return blended, preds, aux

    def save(self, output_dir: str, tag: str):
        for model in self.models.values():
            model.save(output_dir, tag)

    def load(self, output_dir: str, tag: str):
        for model in self.models.values():
            model.load(output_dir, tag)

    def get_model(self, name: str):
        return self.models.get(name)

    def list_model_names(self) -> List[str]:
        return list(self.models.keys())

