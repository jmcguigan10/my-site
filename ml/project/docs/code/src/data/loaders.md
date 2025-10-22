# src/data/loaders.py

[Home](../../../index.md) · [Site Map](../../../site-map.md) · [Code Browser](../../../code-browser.md) · [Folder README](../../../../src/data/README.md)

**Open original file:** [loaders.py](../../../../src/data/loaders.py)

## Preview

```python
from typing import Tuple
import numpy as np

def make_synthetic_linear(n_samples: int = 512, n_features: int = 3, noise_std: float = 0.1, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    w_true = rng.normal(size=(n_features, 1))
    b_true = float(rng.normal())
    y = X @ w_true + b_true + rng.normal(scale=noise_std, size=(n_samples, 1))
    return X, y, w_true.flatten(), b_true

```
