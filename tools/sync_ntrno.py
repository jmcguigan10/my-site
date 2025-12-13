#!/usr/bin/env python3
"""
Sync selected artifacts from the ntrno_stblty_clssfr repo into the site.

- copies README.md and renders it to HTML for embedding
- copies plot assets into assets/ntrno/
- writes a small metadata JSON with commit + updated_at + plot list
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
NTRNO_REPO = ROOT / "external" / "ntrno"
PAGE_DIR = ROOT / "ml" / "ntrno"
ASSETS_DIR = ROOT / "assets" / "ntrno"


def _ensure_paths() -> None:
    if not NTRNO_REPO.exists():
        sys.exit("ntrno submodule missing; run `git submodule update --init`.")
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "loss").mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "f1_t_sweep").mkdir(parents=True, exist_ok=True)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _md_to_html(markdown_text: str) -> str:
    try:
        import markdown  # type: ignore
    except Exception:
        # Fallback: basic preformatted block.
        return f"<pre>{_html_escape(markdown_text)}</pre>"

    return markdown.markdown(  # type: ignore[attr-defined]
        markdown_text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )


def _copy_readme() -> None:
    src = NTRNO_REPO / "README.md"
    md = src.read_text(encoding="utf-8")
    (PAGE_DIR / "README.md").write_text(md, encoding="utf-8")
    html = _md_to_html(md)
    (PAGE_DIR / "readme.html").write_text(html, encoding="utf-8")


def _copy_plots() -> List[Dict[str, str]]:
    plot_meta: List[Dict[str, str]] = []
    sources = {
        "loss": NTRNO_REPO / "assets" / "plots" / "loss",
        "f1_t_sweep": NTRNO_REPO / "assets" / "plots" / "f1_t_sweep",
    }

    for key, src_dir in sources.items():
        if not src_dir.exists():
            continue
        dest_dir = ASSETS_DIR / key
        for png in sorted(src_dir.glob("*.png")):
            dest_path = dest_dir / png.name
            shutil.copy2(png, dest_path)
            plot_meta.append(
                {
                    "kind": key,
                    "filename": png.name,
                    "relative": f"../../assets/ntrno/{key}/{png.name}",
                }
            )
    return plot_meta


def _git_info() -> Dict[str, str]:
    def _run(args: List[str]) -> str:
        return (
            subprocess.check_output(args, cwd=NTRNO_REPO)
            .decode("utf-8")
            .strip()
        )

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _write_meta(plots: List[Dict[str, str]], git_info: Dict[str, str]) -> None:
    meta = {
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info,
        "plots": plots,
    }
    (PAGE_DIR / "sync-meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def main() -> None:
    _ensure_paths()
    _copy_readme()
    plots = _copy_plots()
    info = _git_info()
    _write_meta(plots, info)
    print(
        f"Synced ntrno @ {info['short']} with {len(plots)} plots "
        f"into {PAGE_DIR} and {ASSETS_DIR}"
    )


if __name__ == "__main__":
    main()
