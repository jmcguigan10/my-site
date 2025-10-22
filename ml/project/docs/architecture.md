> Navigation: [Home](index.md) · [Site Map](site-map.md) · [Code Browser](code-browser.md)

<style>
  .back-link {
    position: fixed;       /* stay in viewport as you scroll */
    top: 10px;              /* adjust vertical offset */
    left: 10px;             /* adjust horizontal offset */
    z-index: 1000;          /* sit above most stuff */
  }

  .back-link img {
    width: 30px;             /* size of your arrow — change as needed */
    height: auto;
    cursor: pointer;
  }
</style>

<a href="../README.md" class="back-link">
  <img src="back-arrow.png" alt="Back">
</a>

# System Architecture

This markdown is the "web" of links between pieces. Start anywhere, follow the trails.

- `configs/*.yaml` → drives `scripts/run_training.py`
- `scripts/run_training.py` → loads YAML via `src/utils/config.py`, sets seeds and logging
- `src/training/train.py` → orchestrates data → model → optimizer
- `src/algorithms/*` → the optimizers selected by `training.algorithm`
- `src/models/linear_regression.py` → the example model
- `src/data/loaders.py` → synthetic data generator

## Graph (Mermaid)

```mermaid
graph TD
  C[configs/*.yaml] --> R[scripts/run_training.py]
  R --> U[src/utils/config.py]
  R --> T[src/training/train.py]
  T --> D[src/data/loaders.py]
  T --> M[src/models/linear_regression.py]
  T --> A[src/algorithms/*]
```

- See also: [Directory Map](directory-map.md)