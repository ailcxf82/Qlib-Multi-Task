"""
简易持久化工具，负责保存与加载模型中间产物。
"""

from __future__ import annotations

import os
import pickle
from typing import Any


def ensure_dir(path: str):
    """确保持久化路径存在。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_pickle(obj: Any, path: str):
    """持久化对象到 pickle。"""
    ensure_dir(path)
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path: str) -> Any:
    """从 pickle 载入对象。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "rb") as fp:
        return pickle.load(fp)


