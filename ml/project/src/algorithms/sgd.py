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
