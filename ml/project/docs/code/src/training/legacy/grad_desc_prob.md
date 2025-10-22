# src/training/legacy/grad_desc_prob.py

[Home](../../../../index.md) · [Site Map](../../../../site-map.md) · [Code Browser](../../../../code-browser.md)

**Open original file:** [grad_desc_prob.py](../../../../../src/training/legacy/grad_desc_prob.py)

## Preview

```python
# "Soft Safe" Gradient Descent Playground
# ------------------------------------------------------------
# This single Python cell gives you:
#   • A smooth nonconvex loss on a 3‑torus (three circular dials).
#   • Multiple optimizers: fixed-step GD, backtracking GD, momentum/Nesterov, Adam.
#   • Options for learning‑rate schedules, noise, multi‑start, and diagnostics.
#   • Heavy comments with the math, when to use which option, and why.
#
# RULES (for this environment):
#   - Uses Matplotlib (no seaborn), one chart per figure, no custom colors/styles.
#
# You can scroll through the code comments as a mini‑tutorial.
# At the bottom, we run a few demo experiments and draw simple plots.
#
# Notation:
#   θ ∈ T^3 where T is a circle; each θ_i is wrapped to [-π, π).
#   d(θ, θ*) = wrap(θ - θ*) is the shortest signed angular difference.
#
# GLOBAL OBJECTS created:
#   - theta_star: the hidden safe combination (randomized per run).
#   - couplings: dict with c1,c2,c3 shaping nonconvexity.
#
# ------------------------------------------------------------

import numpy as np
import math
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)  # change seed to reshuffle the secret combo each run

# --------------------------
# Geometry utilities on the torus T^3
# --------------------------
def wrap_angle(a):
    """
    Wrap a (possibly vector) angle to [-π, π).
    This defines the equivalence relation of angles modulo 2π and
    is essential for optimization on a torus rather than R^3.
    """
    return (a + np.pi) % (2*np.pi) - np.pi

def angdiff(a, b):
    """
    Shortest signed angular difference in [-π, π):
    angdiff(a,b) = wrap(a - b).
    """
    return wrap_angle(a - b)

def torus_distance(a, b):
    """
    L2 distance on the torus: ||wrap(a-b)||_2.
    This respects periodicity and is the relevant notion of "closeness."
    """
    d = angdiff(np.asarray(a), np.asarray(b))
    return float(np.linalg.norm(d))

# --------------------------
# Loss: "Soft Safe" with gentle couplings
# --------------------------
# Global minimum is at θ = θ* (mod 2π). The base term is convex on each angle,
# but couplings introduce soft nonconvex structure and benign local minima.
theta_star = rng.uniform(-np.pi, np.pi, size=3)  # secret target
couplings = dict(c1=0.30, c2=0.20, c3=0.10)

def lock_loss(theta, theta_star=theta_star, couplings=couplings):
    r"""
    L(θ) = Σ_i (1 - cos d_i)
           + c1 * (1 - cos(d0 + 2 d1))
           + c2 * (1 - cos(2 d1 - d2))
           + c3 * (1 - cos(3 d0 - d1 + d2))
    where d = wrap(θ - θ*).

    WHY THIS FORM:
    - 1 - cos(d_i) has a unique minimum at d_i = 0, quadratic near 0:
        1 - cos(d_i) ≈ 0.5 d_i^2.
      Its gradient is sin(d_i). This creates a bowl around θ*.
    - Coupling terms are low‑weight perturbations that keep the same global
      minimizer but create gentle local basins so GD has to think.
    """
    theta = np.asarray(theta, dtype=float)
    d = angdiff(theta, theta_star)
    base = float(np.sum(1 - np.cos(d)))
    c1, c2, c3 = couplings["c1"], couplings["c2"], couplings["c3"]
    t1 = 1 - np.cos(d[0] + 2*d[1])
    t2 = 1 - np.cos(2*d[1] - d[2])
    t3 = 1 - np.cos(3*d[0] - d[1] + d[2])
    return base + c1*t1 + c2*t2 + c3*t3

def lock_grad(theta, theta_star=theta_star, couplings=couplings):
    r"""
    Analytic gradient ∇L(θ). Using d = wrap(θ - θ*).
    For a term 1 - cos(u), d/du = sin(u).
    Chain rule gives ∂/∂θ_i [1 - cos(d_i)] = sin(d_i) * ∂d_i/∂θ_i.
    Ignoring the measure‑zero wrap kinks, ∂d_i/∂θ_i = 1 almost everywhere.

    Coupling terms:
      t1 = 1 - cos(d0 + 2 d1)     ⇒ ∇ = sin(d0 + 2 d1) * [1, 2, 0]
      t2 = 1 - cos(2 d1 - d2)     ⇒ ∇ = sin(2 d1 - d2) * [0, 2, -1]
      t3 = 1 - cos(3 d0 - d1 + d2)⇒ ∇ = sin(3 d0 - d1 + d2) * [3, -1, 1]

    NOTE: On the torus, take a gradient step in R^3, then wrap.
    """
    theta = np.asarray(theta, dtype=float)
    d = angdiff(theta, theta_star)
    g = np.zeros(3, dtype=float)

    # base
    g += np.sin(d)

    c1, c2, c3 = couplings["c1"], couplings["c2"], couplings["c3"]

    s1 = np.sin(d[0] + 2*d[1])
    g[0] += c1 * s1 * 1.0
    g[1] += c1 * s1 * 2.0

    s2 = np.sin(2*d[1] - d[2])
    g[1] += c2 * s2 * 2.0
    g[2] += c2 * s2 * (-1.0)

    s3 = np.sin(3*d[0] - d[1] + d[2])
    g[0] += c3 * s3 * 3.0
    g[1] += c3 * s3 * (-1.0)
    g[2] += c3 * s3 * 1.0

    return g

# --------------------------
# Gradient checker (optional but educational)
# --------------------------
def check_grad(f, grad, x, h=1e-5):
    """
    Central finite‑difference gradient check on the torus.
    Returns relative error ||g_fd - g|| / max(1, ||g||, ||g_fd||).

    Use this once to sanity‑check your analytic gradient.
    """
    x = np.asarray(x, dtype=float)
    g_fd = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = 1.0
        xph = wrap_angle(x + h*e)
        xmh = wrap_angle(x - h*e)
        g_fd[i] = (f(xph) - f(xmh)) / (2*h)
    g = grad(x)
    denom = max(1.0, float(np.linalg.norm(g)), float(np.linalg.norm(g_fd)))
    rel = float(np.linalg.norm(g_fd - g)) / denom
    return rel, g, g_fd

# --------------------------
# Learning‑rate schedules
# --------------------------
def lr_constant(lr0):
    """Constant LR: η_k = lr0. Good when a line search is expensive and the landscape is tame."""
    def s(k): return lr0
    return s

def lr_geometric(lr0, gamma=0.99):
    """Geometric decay: η_k = lr0 * γ^k. Useful when you want coarse steps early, fine steps late."""
    def s(k): return lr0 * (gamma ** k)
    return s

def lr_inverse_sqrt(lr0, warmup=10):
    """Inverse‑sqrt: η_k = lr0 / sqrt(k + warmup). A classic schedule with diminishing steps."""
    def s(k): return lr0 / math.sqrt(k + warmup)
    return s

# --------------------------
# Core optimizers on the torus
# --------------------------
def gd_fixed(f, grad, x0, lr_schedule, max_iter=2000, tol_g=1e-9, tol_x=1e-10,
             noise_std=0.0, noise_every=None, callback=None):
    r"""
    Fixed‑step gradient descent on T^3.

    Update:
        g_k = ∇f(x_k)
        η_k = schedule(k)
        x_{k+1} = wrap(x_k - η_k g_k)

    Stopping:
        ||g_k|| < tol_g   or   ||wrap(x_{k+1}-x_k)|| < tol_x   or   k hits max_iter.

    Options:
      - lr_schedule: one of the schedule factories above.
      - noise_std / noise_every: add small Gaussian noise to escape shallow traps.
        x_{k+1} ← wrap(x_{k+1} + N(0, noise_std^2 I)) every 'noise_every' steps.
    """
    x = np.asarray(x0, dtype=float)
    xs = [x.copy()]; fs = [float(f(x))]
    for k in range(max_iter):
        g = grad(x)
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol_g: break
        step = lr_schedule(k)
        xn = wrap_angle(x - step * g)
        if noise_every is not None and ((k+1) % noise_every == 0) and noise_std > 0:
            xn = wrap_angle(xn + rng.normal(0, noise_std, size=xn.shape))
        xs.append(xn.copy()); fs.append(float(f(xn)))
        if float(np.linalg.norm(angdiff(xn, x))) < tol_x: x = xn; break
        x = xn
    return x, np.array(xs), np.array(fs)

def gd_backtracking(f, grad, x0, lr_init=0.3, alpha=1e-4, beta=0.6, max_iter=2000,
                    tol_g=1e-9, tol_x=1e-10, callback=None):
    r"""
    Backtracking line search (Armijo). Robust first choice if you're allergic to tuning.

    Pick step η starting from lr_init, shrink by β ∈ (0,1) until
        f(x - η g) ≤ f(x) - α η ||g||^2
    where α ∈ (0, 0.5) is the sufficient‑decrease parameter.

    WHY:
      - Automatic step control; works well when curvature varies across the space.
      - More function evals per iteration, but fewer faceplants.
    """
    x = np.asarray(x0, dtype=float)
    xs = [x.copy()]; fx = float(f(x)); fs = [fx]
    for k in range(max_iter):
        g = grad(x)
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol_g: break
        step = lr_init
        # Candidate
        xn = wrap_angle(x - step * g)
        fn = float(f(xn))
        # Armijo backtracking
        iters = 0
        while fn > fx - alpha * step * (gnorm**2):
            step *= beta
            xn = wrap_angle(x - step * g)
            fn = float(f(xn))
            iters += 1
            if iters > 60:
                break
        xs.append(xn.copy()); fs.append(fn)
        if float(np.linalg.norm(angdiff(xn, x))) < tol_x:
            x = xn; fx = fn; break
        x = xn; fx = fn
    return x, np.array(xs), np.array(fs)

def gd_momentum(f, grad, x0, lr=0.2, mu=0.9, nesterov=False, max_iter=2000,
                tol_g=1e-9, tol_x=1e-10, callback=None):
    r"""
    Momentum / Nesterov on T^3.

    Classical momentum:
        v_{k+1} = μ v_k - η ∇f(x_k)
        x_{k+1} = wrap(x_k + v_{k+1})

    Nesterov (look‑ahead) gradient:
        g_k = ∇f(x_k + μ v_k)
        v_{k+1} = μ v_k - η g_k
        x_{k+1} = wrap(x_k + v_{k+1})

    WHEN:
      - Use when the landscape has long, gently sloped valleys; momentum accelerates.
      - Nesterov can reduce overshoot by peeking ahead.
    """
    x = np.asarray(x0, dtype=float)
    v = np.zeros_like(x)
    xs = [x.copy()]; fs = [float(f(x))]
    for k in range(max_iter):
        if nesterov:
            g = grad(wrap_angle(x + mu * v))
        else:
            g = grad(x)
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol_g: break
        v = mu * v - lr * g
        xn = wrap_angle(x + v)
        xs.append(xn.copy()); fs.append(float(f(xn)))
        if float(np.linalg.norm(angdiff(xn, x))) < tol_x:
            x = xn; break
        x = xn
    return x, np.array(xs), np.array(fs)

def adam_torus(f, grad, x0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8,
               max_iter=4000, tol_g=1e-9, tol_x=1e-10, callback=None):
    r"""
    Adam optimizer on the torus.

    m_{k+1} = β1 m_k + (1-β1) g_k
    v_{k+1} = β2 v_k + (1-β2) g_k^2
    m̂ = m_{k+1} / (1-β1^{k+1})
    v̂ = v_{k+1} / (1-β2^{k+1})
    x_{k+1} = wrap(x_k - lr * m̂ / (sqrt(v̂) + eps))

    WHEN:
      - Coordinates have different curvature/scales. Adam adaptively rescales steps.
      - Often converges with minimal tuning; beware too‑large lr.
    """
    x = np.asarray(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    xs = [x.copy()]; fs = [float(f(x))]
    for k in range(max_iter):
        g = grad(x)
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol_g: break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1 ** (k + 1))
        vhat = v / (1 - beta2 ** (k + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        xn = wrap_angle(x - step)
        xs.append(xn.copy()); fs.append(float(f(xn)))
        if float(np.linalg.norm(angdiff(xn, x))) < tol_x:
            x = xn; break
        x = xn
    return x, np.array(xs), np.array(fs)

# --------------------------
# Multi‑start driver and success metrics
# --------------------------
def solve_multistart(optimizer_fn, f, grad, n_starts=40, **opt_kwargs):
    """
    Try many random initializations; keep best result and compute success rate.
    A run is 'successful' if it recovers θ* within tight torus distance and loss.
    """
    best = None
    successes = 0
    paths, losses = None, None
    for _ in range(n_starts):
        x0 = rng.uniform(-np.pi, np.pi, size=3)
        x, xs, fs = optimizer_fn(f, grad, x0, **opt_kwargs)
        val = f(x)
        dist = torus_distance(x, theta_star)
        ok = (dist < 1e-3) and (val < 1e-8)
        if ok: successes += 1
        if (best is None) or (val < best[1]):
            best = (x, val, x0)
            paths, losses = xs, fs
    success_rate = successes / n_starts
    return dict(best_x=best[0], best_val=best[1], best_start=best[2],
                best_path=paths, best_losses=losses, success_rate=success_rate)

# --------------------------
# 1) Sanity check the gradient
# --------------------------
x_check = rng.uniform(-np.pi, np.pi, size=3)
rel_err, g_analytic, g_fd = check_grad(lock_loss, lock_grad, x_check)
print("Gradient check at random point")
print("  point:", np.round(x_check, 4))
print("  rel error:", f"{rel_err:.3e}", "(<1e-6 is excellent; <1e-4 is fine for this problem)")

# --------------------------
# 2) Single‑run comparisons from the same start
# --------------------------
x0 = rng.uniform(-np.pi, np.pi, size=3)
print("\nSingle‑run comparisons from a shared start:")
print("  secret θ*:", np.round(theta_star, 6))
print("  start θ0 :", np.round(x0, 6))

# Fixed‑step with geometric decay
x_fixed, path_fixed, loss_fixed = gd_fixed(
    lock_loss, lock_grad, x0,
    lr_schedule=lr_geometric(lr0=0.35, gamma=0.985),
    max_iter=800
)

# Backtracking
x_bt, path_bt, loss_bt = gd_backtracking(
    lock_loss, lock_grad, x0,
    lr_init=0.35, alpha=1e-4, beta=0.6, max_iter=400
)

# Momentum (Nesterov on)
x_mom, path_mom, loss_mom = gd_momentum(
    lock_loss, lock_grad, x0,
    lr=0.22, mu=0.9, nesterov=True, max_iter=800
)

# Adam
x_adam, path_adam, loss_adam = adam_torus(
    lock_loss, lock_grad, x0,
    lr=0.12, max_iter=1200
)

def report(name, x_final, losses):
    dist = torus_distance(x_final, theta_star)
    print(f"  {name:12s} | iters={len(losses)-1:4d}  loss={losses[-1]:.3e}  dist={dist:.3e}")

report("fixed‑step", x_fixed, loss_fixed)
report("backtracking", x_bt, loss_bt)
report("momentum+NAG", x_mom, loss_mom)
report("adam", x_adam, loss_adam)

# Plot loss curves (one figure per optimizer)
plt.figure()
plt.plot(np.arange(len(loss_fixed)), loss_fixed)
plt.title("Fixed‑step GD: loss vs iteration")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(np.arange(len(loss_bt)), loss_bt)
plt.title("Backtracking GD: loss vs iteration")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(np.arange(len(loss_mom)), loss_mom)
plt.title("Momentum (Nesterov): loss vs iteration")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(np.arange(len(loss_adam)), loss_adam)
plt.title("Adam: loss vs iteration")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.show()

# --------------------------
# 3) Success probability vs learning rate (fixed‑step)
# --------------------------
lrs = [0.05, 0.1, 0.18, 0.25, 0.35, 0.5]
success = []
for lr0 in lrs:
    res = solve_multistart(
        gd_fixed, lock_loss, lock_grad, n_starts=30,
        lr_schedule=lr_constant(lr0), max_iter=800
    )
    success.append(res["success_rate"])

plt.figure()
plt.plot(lrs, success, marker='o')
plt.title("Fixed‑step GD: success rate vs learning rate")
plt.xlabel("learning rate")
plt.ylabel("success fraction")
plt.ylim(0, 1.05)
plt.show()

# --------------------------
# 4) Multi‑start with backtracking (robust baseline)
# --------------------------
res_bt = solve_multistart(
    gd_backtracking, lock_loss, lock_grad, n_starts=60,
    lr_init=0.35, alpha=1e-4, beta=0.6, max_iter=600
)
x_best = res_bt["best_x"]
dist_best = torus_distance(x_best, theta_star)
print("\nMulti‑start (backtracking):")
print("  best start:", np.round(res_bt["best_start"], 6))
print("  recovered :", np.round(x_best, 6))
print("  distance to θ*:", f"{dist_best:.3e}")
print("  final loss:", f"{res_bt['best_val']:.3e}")
print("  success rate:", f"{res_bt['success_rate']:.2f} over 60 starts")

plt.figure()
plt.plot(np.arange(len(res_bt["best_losses"])), res_bt["best_losses"])
plt.title("Best run (backtracking): loss vs iteration")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.show()

# --------------------------
# 5) Visual intuition: 2D contour slice with a path overlay
#    Fix θ1 and visualize L(θ0, θ2 | θ1=const). Overlay the best backtracking path.
# --------------------------
theta1_fixed = float(res_bt["best_path"][0][1])  # take initial θ1 for the best run
grid = 150
t0 = np.linspace(-np.pi, np.pi, grid)
t2 = np.linspace(-np.pi, np.pi, grid)
T0, T2 = np.meshgrid(t0, t2, indexing='xy')
Z = np.zeros_like(T0)

# Build the slice; this is compute heavy but fine at 150x150
for i in range(grid):
    for j in range(grid):
        th = np.array([T0[i, j], theta1_fixed, T2[i, j]])
        Z[i, j] = lock_loss(th)

plt.figure()
# contourf uses a default colormap; we don't specify any styles.
cs = plt.contourf(T0, T2, Z, levels=25)
plt.colorbar()
plt.title(r"Contour slice: $L(\theta_0,\theta_2\,|\,\theta_1=\mathrm{const})$")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_2$")

# Overlay the best path (project to this slice by taking its θ0, θ2)
p = np.array(res_bt["best_path"])
plt.plot(p[:,0], p[:,2], marker='o', linewidth=1)
plt.show()

# --------------------------
# 6) Optional: demonstrate noise annealing escape
# --------------------------
# We'll run a fixed‑step with tiny periodic noise; sometimes this bumps you out of a shallow trap.
x0_noise = rng.uniform(-np.pi, np.pi, size=3)
x_noise, path_noise, loss_noise = gd_fixed(
    lock_loss, lock_grad, x0_noise,
    lr_schedule=lr_constant(0.22),
    max_iter=800,
    noise_std=0.02, noise_every=10
)
print("\nNoise‑annealed fixed‑step example:")
print("  start:", np.round(x0_noise, 6))
print("  end  :", np.round(x_noise, 6))
print("  dist :", f"{torus_distance(x_noise, theta_star):.3e}")
plt.figure()
plt.plot(np.arange(len(loss_noise)), loss_noise)
plt.title("Fixed‑step with periodic noise: loss vs iteration")
plt.xlabel("iteration"); plt.ylabel("loss")
plt.show()
```
