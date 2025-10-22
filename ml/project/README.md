**Docs hub:** [docs/index.md](docs/index.md) · [Site Map](docs/site-map.md) · [Code Browser](docs/code-browser.md)


> Navigation: [Home](docs/index.md) · [Site Map](docs/site-map.md) · [Code Browser](docs/code-browser.md)

# ML Optimization Project

A clean, modular scaffold carved out of a single-file script. Toggle between learning algorithms via YAML. Run experiments without the spaghetti.

## TL;DR

1. Choose a config in `configs/` (e.g. `configs/adam.yaml`).
2. Run:
   ```bash
   python scripts/run_training.py --config configs/adam.yaml
   ```

Your original file has been preserved at `src/training/legacy/grad_desc_prob.py`.

## Directory tree

```text
.
├── configs/
│   ├── default.yaml
│   ├── sgd.yaml
│   ├── momentum.yaml
│   └── adam.yaml
``` 
[README](configs/README.md)
```text
├── docs/
│   ├── index.md
│   ├── architecture.md
│   └── directory-map.md
``` 
- [Architecture](docs/architecture.md)
- [Directory](docs/directory-map.md)
- [Index](docs/index.md)
```text
├── experiments/
│   └── 2025-10-20-baseline/
│       ├── notes.md
│       └── .gitkeep
``` 
[README](experiments/README.md)
```text
├── scripts/
│   └── run_training.py
``` 
[README](scripts/README.md)
```text
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── sgd.py
│   │   ├── momentum.py
│   │   └── adam.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── linear_regression.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── legacy/
│   │       └── grad_desc_prob.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
``` 
[README](src/README.md)
```text
└── tests/
    └── test_optimizers.py
```

## Add a new algorithm

- Create a new file under `src/algorithms/your_algo.py` implementing a class with `.step(params, grads)` and an optional `.state` dict.
- Add a new config file under `configs/your_algo.yaml` with `training.algorithm: "your_algo"`.
- Run with `python scripts/run_training.py --config configs/your_algo.yaml`.

See `docs/architecture.md` for the Markdown link web and system diagram.