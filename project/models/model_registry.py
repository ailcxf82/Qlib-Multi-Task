"""
模型注册表：集中管理可用模型，便于在 Ensemble 中动态扩展。
"""

from __future__ import annotations

from typing import Dict, Type

from models.lightgbm_model import LightGBMModelWrapper
from models.mlp_model import MLPRegressor
# NOTE: 若后续需要引入更多模型，在此注册即可。
MODEL_REGISTRY: Dict[str, Type] = {
    "lightgbm": LightGBMModelWrapper,
    "mlp": MLPRegressor,
}


def create_model(model_type: str, config_path: str):
    """根据类型创建模型实例。"""
    cls = MODEL_REGISTRY.get(model_type.lower())
    if cls is None:
        raise ValueError(f"未注册的模型类型: {model_type}")
    return cls(config_path)

