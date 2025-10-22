> Navigation: [Home](../docs/index.md) · [Site Map](../docs/site-map.md) · [Code Browser](../docs/code-browser.md)

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

## Configs

Pick one of these YAMLs when running `scripts/run_training.py`. The `training.algorithm` key selects which optimizer to use.

- `default.yaml` minimal example
- `sgd.yaml` stochastic gradient descent
- `momentum.yaml` SGD with momentum
- `adam.yaml` Adam optimizer

You can clone one and change hyperparameters as needed.