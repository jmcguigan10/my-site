# src/utils/logging.py

[Home](../../../index.md) · [Site Map](../../../site-map.md) · [Code Browser](../../../code-browser.md) · [Folder README](../../../../src/utils/README.md)

**Open original file:** [logging.py](../../../../src/utils/logging.py)

## Preview

```python
import logging
from pathlib import Path
from typing import Optional

def get_logger(name: str = "app", level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers in notebooks

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.setLevel(numeric_level)
    logger.addHandler(stream)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    return logger

```
