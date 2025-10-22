# src/training/train.py

[Home](../../../index.md) · [Site Map](../../../site-map.md) · [Code Browser](../../../code-browser.md) · [Folder README](../../../../src/training/README.md)

**Open original file:** [train.py](../../../../src/training/train.py)

## Preview

```python
from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.data.loaders import make_synthetic_linear
from src.models.linear_regression import LinearRegressionModel
from src.algorithms import SGD, SGDMomentum, Adam

class Trainer:
    def __init__(self, cfg: Dict[str, Any], logger, out_dir: Path):
        self.cfg = cfg
        self.logger = logger
        self.out_dir = out_dir

        data_cfg = cfg.get("data", {})
        exp_cfg = cfg.get("experiment", {})
        seed = int(exp_cfg.get("seed", 0))

        X, y, w_true, b_true = make_synthetic_linear(
            n_samples=int(data_cfg.get("n_samples", 512)),
            n_features=int(data_cfg.get("n_features", 3)),
            noise_std=float(data_cfg.get("noise_std", 0.1)),
            seed=seed,
        )
        self.X, self.y = X, y
        self.w_true, self.b_true = w_true, b_true

        model_cfg = cfg.get("model", {})
        if model_cfg.get("type", "linear_regression") != "linear_regression":
            raise ValueError("Only 'linear_regression' model is implemented in this scaffold.")
        self.model = LinearRegressionModel(n_features=int(data_cfg.get("n_features", 3)))

        tr = cfg.get("training", {})
        algo = tr.get("algorithm", "sgd").lower()
        lr = float(tr.get("lr", 0.01))

        if algo == "sgd":
            self.opt = SGD(lr=lr)
        elif algo == "momentum":
            self.opt = SGDMomentum(lr=lr, momentum=float(tr.get("momentum", 0.9)))
        elif algo == "adam":
            self.opt = Adam(lr=lr, beta1=float(tr.get("beta1", 0.9)), beta2=float(tr.get("beta2", 0.999)), eps=float(tr.get("eps", 1e-8)))
        else:
            raise ValueError(f"Unknown training.algorithm: {algo}")

        self.epochs = int(tr.get("epochs", 100))
        self.batch_size = int(tr.get("batch_size", 64))

    def batches(self, X, y, batch_size):
        n = X.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            j = idx[i:i+batch_size]
            yield X[j], y[j]

    def run(self):
        self.logger.info("Starting training")
        history = []
        for epoch in range(1, self.epochs + 1):
            for Xb, yb in self.batches(self.X, self.y, self.batch_size):
                loss, grads = self.model.loss_and_grads(Xb, yb)
                self.model.params = self.opt.step(self.model.params, grads)

            # full pass for logging
            loss_full, _ = self.model.loss_and_grads(self.X, self.y)
            history.append({"epoch": epoch, "loss": loss_full})
            if epoch % max(1, self.epochs // 10) == 0 or epoch == 1:
                self.logger.info(f"epoch {epoch:3d} | loss={loss_full:.6f}")

        # Save run artifacts
        np.savez(self.out_dir / "artifacts.npz", w=self.model.params["w"], b=self.model.params["b"], w_true=self.w_true, b_true=self.b_true)
        with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
            import json
            json.dump(history, f, indent=2)
        self.logger.info("Training complete. Artifacts written to %s", str(self.out_dir))

```
