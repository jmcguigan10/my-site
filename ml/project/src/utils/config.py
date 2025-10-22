from pathlib import Path
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config. Falls back with a clear message if PyYAML isn't installed."""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load configs. Install with `pip install pyyaml`."
        ) from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping (dict)." )
    return data
