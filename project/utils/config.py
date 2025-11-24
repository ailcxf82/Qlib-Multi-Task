"""
配置加载工具，负责解析 YAML 并提供统一的字典对象。
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    参数
    ----
    path : str
        文件相对或绝对路径。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


