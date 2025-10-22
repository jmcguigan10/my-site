# src/models/linear_regression.py

[Home](../../../index.md) · [Site Map](../../../site-map.md) · [Code Browser](../../../code-browser.md) · [Folder README](../../../../src/models/README.md)

**Open original file:** [linear_regression.py](../../../../src/models/linear_regression.py)

## Preview

```python
from typing import Dict
import numpy as np

class LinearRegressionModel:
    def __init__(self, n_features: int):
        self.params = {
            "w": np.zeros((n_features, 1)),
            "b": np.zeros((1, 1)),
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.params["w"] + self.params["b"]

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> (float, Dict[str, np.ndarray]):
        # Mean squared error
        y_pred = self.forward(X)
        err = y_pred - y
        loss = float((err ** 2).mean())
        n = X.shape[0]
        grads = {
            "w": (2.0 / n) * X.T @ err,
            "b": (2.0 / n) * err.sum(keepdims=True)
        }
        return loss, grads

```
