
# FFI Stability Classifier — Training Pack

This bundle gives you a single-command training pipeline that:
- trains **two models** (seeds `17` and `43`) on your NPZ datasets,
- logs accuracy/precision/recall/F1 each epoch,
- prints the summary line exactly like you asked,
- makes **loss** and **F1-vs-threshold** plots,
- measures **inference timings** over lots of points and stores the per-sample timestamps to CSV,
- exports **TorchScript** and **ONNX** for C++ consumption,
- writes everything to neat folders so future-you doesn’t curse present-you.

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

```
ffi_stability_project/
├── data/                         # Put your NPZ training files here
│   ├── train_data_NSM_stable.npz
│   ├── train_data_random.npz
│   ├── train_data_stable_oneflavor.npz
│   └── train_data_stable_zerofluxfac.npz
├── slurm/
│   └── auto_model.slurm          # Run this on the cluster
├── src/
│   ├── full_nn.py                # Architecture + training + export + plotting
│   └── train_two_models.py       # Small wrapper that launches seeds 17 & 43
├── models/
│   ├── pytorch/                  # .pt state_dicts
│   ├── torchscript/              # .ts files
│   └── onnx/                     # .onnx files
├── outputs/
│   ├── logs/                     # per-run .txt logs
│   ├── inference/                # inference_timestamps_*.csv
│   └── plots/
│       ├── loss_curves/          # loss_seed*.png
│       └── f1_threshold/         # f1_vs_threshold_seed*.png
└── README.md
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
