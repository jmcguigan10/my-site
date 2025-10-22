# src/algorithms/sgd.py

[Home](../../../index.md) · [Site Map](../../../site-map.md) · [Code Browser](../../../code-browser.md) · [Folder README](../../../../src/algorithms/README.md)

**Open original file:** [sgd.py](../../../../src/algorithms/sgd.py)

## Preview

```python
from typing import Dict, Any, Tuple
import numpy as np

class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.state: Dict[str, Any] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for k in params:
            params[k] = params[k] - self.lr * grads[k]
        return params

```
