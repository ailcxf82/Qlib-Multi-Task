"""
运行训练流程：加载配置 -> 滚动训练 -> 保存模型。
"""

import argparse
import logging
import os

from trainer.trainer import RollingTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib 因子模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="pipeline 配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    trainer = RollingTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()


