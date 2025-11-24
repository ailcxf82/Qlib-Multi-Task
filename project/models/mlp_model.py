"""
PyTorch MLP 回归模型，实现基础训练、验证与持久化。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_yaml_config

logger = logging.getLogger(__name__)


def _build_mlp(input_dim: int, hidden_dims: List[int], dropout: float, activation: str) -> nn.Module:
    """根据配置构造 MLP。"""
    act_cls = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }.get(activation.lower(), nn.ReLU)
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


class MLPRegressor:
    """MLP 模型封装。"""

    def __init__(self, config: Union[str, Dict[str, Any]]):
        if isinstance(config, str):
            raw = load_yaml_config(config)
        else:
            raw = config
        self.config = raw.get("model", raw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None

    def _tensorize(self, feat: pd.DataFrame, label: Optional[pd.Series] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 直接将 pandas 数据转换为 tensor，便于组建 TensorDataset
        x = torch.tensor(feat.values, dtype=torch.float32)
        y = torch.tensor(label.values, dtype=torch.float32).unsqueeze(-1) if label is not None else None
        return x, y

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        input_dim = train_feat.shape[1]
        self._input_dim = train_feat.shape[1]
        self.model = _build_mlp(
            input_dim=self._input_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64]),
            dropout=self.config.get("dropout", 0.1),
            activation=self.config.get("activation", "relu"),
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-3),
            weight_decay=self.config.get("weight_decay", 0.0),
        )

        batch_size = self.config.get("batch_size", 1024)
        max_epochs = self.config.get("max_epochs", 20)
        patience = self.config.get("patience", 5)
        # 提前停止监控
        best_loss = float("inf")
        wait = 0
        best_state = None

        train_loader = DataLoader(
            TensorDataset(*self._tensorize(train_feat, train_label)),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = None
        if valid_feat is not None and valid_label is not None and len(valid_feat) > 0 and len(valid_label) > 0:
            valid_loader = DataLoader(
                TensorDataset(*self._tensorize(valid_feat, valid_label)),
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            valid_loader = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            epoch_loss /= len(train_loader.dataset)

            val_loss = epoch_loss
            if valid_loader is not None:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for vx, vy in valid_loader:
                        vx = vx.to(self.device)
                        vy = vy.to(self.device)
                        pred = self.model(vx)
                        total += criterion(pred, vy).item() * len(vx)
                val_loss = total / len(valid_loader.dataset)

            logger.info("MLP epoch %d train_loss=%.6f valid_loss=%.6f", epoch, epoch_loss, val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                best_state = self.model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    logger.info("早停触发，最佳验证损失 %.6f", best_loss)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, feat: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("MLP 模型尚未训练")
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(feat.values, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
        return pd.Series(preds, index=feat.index, name="mlp_pred")

    def save(self, output_dir: str, model_name: str):
        if self.model is None:
            raise RuntimeError("无可保存模型")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_name}_mlp.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )
        meta = {
            "input_dim": self._input_dim,
        }
        with open(os.path.join(output_dir, f"{model_name}_mlp_meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        logger.info("MLP 模型已保存: %s", path)

    def load(self, output_dir: str, model_name: str, input_dim: Optional[int] = None):
        path = os.path.join(output_dir, f"{model_name}_mlp.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if input_dim is None:
            meta_path = os.path.join(output_dir, f"{model_name}_mlp_meta.json")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(meta_path)
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            input_dim = meta["input_dim"]
        self._input_dim = input_dim
        self.model = _build_mlp(
            input_dim=input_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64]),
            dropout=self.config.get("dropout", 0.1),
            activation=self.config.get("activation", "relu"),
        ).to(self.device)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info("MLP 模型已加载: %s", path)

