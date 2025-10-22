from typing import Dict, Any
import numpy as np

class SGDMomentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for k in params:
            v = self.velocity.get(k, np.zeros_like(params[k]))
            v = self.momentum * v - self.lr * grads[k]
            self.velocity[k] = v
            params[k] = params[k] + v
        return params
