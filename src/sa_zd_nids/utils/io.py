"""Atomic I/O helpers for JSON, CSV, JSONL, and model serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def save_model_atomic(path: str | Path, model: Any, method: str = "joblib") -> None:
    """Save model atomically. method: 'joblib'|'torch'|'xgb'."""
    import os
    import tempfile

    ensure_parent(path)
    path = Path(path)
    dirp = path.parent
    if method == "torch":
        import torch

        fd, tmp = tempfile.mkstemp(dir=str(dirp), suffix=".pt")
        os.close(fd)
        torch.save(model, tmp)
        os.replace(tmp, str(path))
        return

    if method == "xgb":
        # model expected to have .save_model API (XGBClassifier)
        fd, tmp = tempfile.mkstemp(dir=str(dirp), suffix=".model")
        os.close(fd)
        try:
            model.save_model(tmp)
            os.replace(tmp, str(path))
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return

    # default joblib
    import joblib

    fd, tmp = tempfile.mkstemp(dir=str(dirp), suffix=".pkl")
    os.close(fd)
    joblib.dump(model, tmp)
    os.replace(tmp, str(path))


def save_metadata_atomic(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write metadata atomically and include version/timestamp if present in payload."""
    import os
    import tempfile

    ensure_parent(path)
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
    os.replace(tmp, str(path))
