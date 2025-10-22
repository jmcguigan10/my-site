import numpy as np
from src.algorithms import SGD, SGDMomentum, Adam

def quad_loss_and_grad(x):
    # f(x) = (x-3)^2
    f = float((x - 3.0) ** 2)
    g = 2.0 * (x - 3.0)
    return f, g

def run_optimizer(opt, steps=200, lr=0.05):
    x = 0.0
    hist = []
    for _ in range(steps):
        f, g = quad_loss_and_grad(x)
        x = opt.step({"x": np.array([[x]])}, {"x": np.array([[g]])})["x"][0,0]
        hist.append(f)
    return hist

def test_sgd_decreases_loss():
    opt = SGD(lr=0.05)
    hist = run_optimizer(opt)
    assert hist[-1] < hist[0]

def test_momentum_decreases_loss():
    opt = SGDMomentum(lr=0.05, momentum=0.9)
    hist = run_optimizer(opt)
    assert hist[-1] < hist[0]

def test_adam_decreases_loss():
    opt = Adam(lr=0.05)
    hist = run_optimizer(opt)
    assert hist[-1] < hist[0]
