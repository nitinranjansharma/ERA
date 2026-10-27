"""Fetch Session 14 illustration images so Colab and a fresh checkout can display them.

The files live in the public ERA S14 assignment folder. They are *session
illustrations* (a language-model dense vs upcycled vs scratch figure), not the
metrics of this synthetic-regression run. This helper downloads them into
``images/`` when they are missing.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

SESSION_IMAGE_URLS: dict[str, str] = {
    "results.png": (
        "https://raw.githubusercontent.com/vedapragna-cs/era5-assignments/"
        "main/S14/images/results.png"
    ),
}

DEFAULT_IMAGE_DIR = "images"


def fetch_session_figures(
    dest_dir: str | Path = DEFAULT_IMAGE_DIR,
    urls: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    overwrite: bool = False,
) -> dict[str, dict[str, str | bool]]:
    """Download missing session figures. Existing files are left alone unless ``overwrite``.

    Returns a per-file report with path, source URL, and whether a download ran.
    """
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mapping = dict(SESSION_IMAGE_URLS if urls is None else urls)
    report: dict[str, dict[str, str | bool]] = {}
    for name, url in mapping.items():
        path = destination / name
        existed = path.is_file() and path.stat().st_size > 0
        downloaded = False
        error = ""
        if overwrite or not existed:
            try:
                urllib.request.urlretrieve(url, path)
                downloaded = path.is_file() and path.stat().st_size > 0
                if not downloaded:
                    error = "download produced an empty file"
            except (urllib.error.URLError, OSError) as exc:
                error = str(exc)
        report[name] = {
            "path": str(path),
            "url": url,
            "exists": path.is_file() and path.stat().st_size > 0,
            "downloaded": downloaded,
            "error": error,
        }
    return report


def session_figure_path(name: str, dest_dir: str | Path = DEFAULT_IMAGE_DIR) -> Path:
    return Path(dest_dir) / name
