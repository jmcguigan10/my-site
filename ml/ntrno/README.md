
# FFI Stability Classifier

## Pipeline Summary
- loads 4 NPZ datasets,
- concatenates them into one pool,
- optionally **upweights the “random” slice** via per-sample weights,
- trains a simple MLP with early stopping + LR warmup + plateau scheduler,
- logs **loss + accuracy/precision/recall/F1** each epoch (at threshold 0.5),
- sweeps thresholds 0.01..0.99 to find the **best F1 threshold**,
- saves a model `.pt`, plots, and a combined `results.npy`,
- benchmarks inference latency on CPU (and GPU if available).

## Directory layout

```text
.
├─ assets/
│  └─ plots/                   # Example plots committed to git
├─ config/
│  ├─ train_example.yaml        # Example training config (EDIT data_dir!)
│  └─ slurm_example.yaml        # Example Slurm config template
├─ hpc/
│  └─ train.slurm               # Generic Slurm wrapper (expects env vars)
├─ scripts/
│  └─ submit_slurm.py           # Builds sbatch command from YAML
├─ src/
│  └─ ntrno/
│     ├─ __init__.py            # PROJECT_ROOT resolution
│     ├─ cli/train.py           # main entrypoint: python -m ntrno.cli.train
│     ├─ config.py              # dataclasses + defaults
│     ├─ data.py                # NPZ loading, weights, split, scaling, loaders
│     ├─ inference.py           # inference latency benchmark helper
│     ├─ metrics.py             # threshold sweep + PR/F1 utilities
│     ├─ models.py              # MLP classifier
│     ├─ plots.py               # plot writers (dark/cyber styling)
│     └─ train.py               # training loop + saving artifacts
├─ tests/
│  └─ test_train_smoke.py       # tiny synthetic smoke test
├─ Makefile
├─ requirements.txt
└─ README.md
```
## Data you must provide
By default, the trainer expects a data/ (found at https://zenodo.org/records) directory at the repo root containing exactly these four files:

```text
data/
├─ train_data_stable_zerofluxfac.npz
├─ train_data_stable_oneflavor.npz
├─ train_data_random.npz
└─ train_data_NSM_stable.npz
```
### Expected NPZ keys
Each NPZ must contain the following arrays:
```
| File                              | Features key    | Labels key             |
| --------------------------------- | --------------- | ---------------------- |
| train_data_stable_zerofluxfac.npz | `X_zerofluxfac` | `unstable_zerofluxfac` |
| train_data_stable_oneflavor.npz   | `X_oneflavor`   | `unstable_oneflavor`   |
| train_data_random.npz             | `X_random`      | `unstable_random`      |
| train_data_NSM_stable.npz         | `X_NSM_stable`  | `unstable_NSM_stable`  |
```

## Makefile shortcuts
The Makefile wires everything up for you:

```bash
make venv
make train
make test
make slurm
```

## Plots at a glance

Loss curves (kept just these two):

<div style="display:flex; gap:12px; justify-content:center; align-items:center; flex-wrap:wrap;">
  <img src="assets/plots/loss/S17_L5_HS384_DR0.01_WD0.0001_BS6144_RM3_loss.png" alt="Seed 17 loss" style="max-width:48%; min-width:240px; width:48%;">
  <img src="assets/plots/loss/S43_L5_HS384_DR0.01_WD0.0001_BS6144_RM3_loss.png" alt="Seed 43 loss" style="max-width:48%; min-width:240px; width:48%;">
</div>

F1 vs threshold sweeps:

<div style="display:flex; gap:12px; justify-content:center; align-items:center; flex-wrap:wrap;">
  <img src="assets/plots/f1_t_sweep/S43_L5_HS384_DR0.01_WD0.0001_BS6144_RM3_f1_sweep.png" alt="Seed 43 sweep" style="max-width:48%; min-width:240px; width:48%;">
  <img src="assets/plots/f1_t_sweep/S17_L6_HS256_DR0.01_WD0.0001_BS4192_RM3_f1_sweep.png" alt="Seed 17 sweep" style="max-width:48%; min-width:240px; width:48%;">
</div>

## Directory layout

```text
.
├─ assets/
│  └─ plots/                   # Example plots committed to git
├─ config/
│  ├─ train_example.yaml        # Example training config (EDIT data_dir!)
│  └─ slurm_example.yaml        # Example Slurm config template
├─ hpc/
│  └─ train.slurm               # Generic Slurm wrapper (expects env vars)
├─ scripts/
│  └─ submit_slurm.py           # Builds sbatch command from YAML
├─ src/
│  └─ ntrno/
│     ├─ __init__.py            # PROJECT_ROOT resolution
│     ├─ cli/train.py           # main entrypoint: python -m ntrno.cli.train
│     ├─ config.py              # dataclasses + defaults
│     ├─ data.py                # NPZ loading, weights, split, scaling, loaders
│     ├─ inference.py           # inference latency benchmark helper
│     ├─ metrics.py             # threshold sweep + PR/F1 utilities
│     ├─ models.py              # MLP classifier
│     ├─ plots.py               # plot writers (dark/cyber styling)
│     └─ train.py               # training loop + saving artifacts
├─ tests/
│  └─ test_train_smoke.py       # tiny synthetic smoke test
├─ Makefile
├─ requirements.txt
└─ README.md
```

## Quick start (local)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m ntrno.cli.train --data-dir ./data --outputs-dir ./outputs2
```

Run a quick smoke test (tiny synthetic data):
```
PYTHONPATH=src python -m unittest tests.test_train_smoke
```

Makefile shortcuts:
```
make venv    # create virtualenv using PYTHON (default python3)
make train   # uses config/train_example.yaml and outputs2/
make test    # smoke test
make slurm   # submit via scripts/submit_slurm.py (see config/slurm_example.yaml -> .env/slurm.yaml)
```

Configs:
- `config/train_example.yaml`: grid/local defaults (copy/edit or pass your own with `--config`).
- `config/slurm_example.yaml`: template; put real HPC values in `.env/slurm.yaml` (ignored) and use `make slurm`.

## On the cluster

Edit the `#SBATCH --chdir` in `slurm/auto_model.slurm` to this folder, then:
```
sbatch slurm/auto_model.slurm
```

### Print format (guaranteed)
Each run ends with a **single-line** summary in the exact format you asked:
```
rm=20 layers=6 hs=384 dr=0.01 wd=0.0001 lr=0.005 bs=32768 | Loss 0.0830 Acc 0.9802 Prec0.5 0.9027 Rec0.5 0.9024 F1@0.5 0.9025 || BEST thr=0.68 Prec 0.9764 Rec 0.8562 F1 0.9124 | Epochs 418 Time 1481.4s
```

### What counts as “C++ compatible”
We export **TorchScript** and **ONNX**. Alongside each model you also get a small JSON with the `mean` and `std` used for feature scaling so your C++ code can mirror preprocessing.

---

## Data format expected
Your NPZ files are autodetected with keys like:
- `X_*` for features of shape `(N, 27)`
- `unstable_*` for labels of shape `(N, 1)` or `(N,)`

We concatenate all found pairs. Labels are treated as `FFI present = 1`.

## Reproduce that exact setup
We lock the architecture to **6 hidden layers**, **hidden size 384**, **dropout 0.01**, and train with:
- `lr=0.005`, `weight_decay=1e-4`
- `batch_size=32768`
- Early stopping patience 50, max epochs 10k
- Class imbalance handled via `pos_weight = (neg/pos)` in BCEWithLogitsLoss

Everything else is just creature comforts.
