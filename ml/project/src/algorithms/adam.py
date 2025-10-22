from typing import Dict, Any
import numpy as np

class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.t += 1
        for k in params:
            g = grads[k]
            m = self.m.get(k, np.zeros_like(g))
            v = self.v.get(k, np.zeros_like(g))

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.m[k] = m
            self.v[k] = v
        return params
