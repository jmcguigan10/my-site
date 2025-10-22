#!/usr/bin/env python
import argparse, os, sys, importlib, random
from pathlib import Path
import numpy as np

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.training.train import Trainer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg.get("experiment", {})
    out_dir = Path(exp.get("output_dir", "experiments/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name="runner", level=cfg.get("logging", {}).get("level", "INFO"), log_dir=cfg.get("logging", {}).get("log_dir", None))

    seed = int(exp.get("seed", 42))
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    trainer = Trainer(cfg, logger=logger, out_dir=out_dir)
    trainer.run()

if __name__ == "__main__":
    main()
