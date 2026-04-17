"""Unit tests for I/O utilities."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sa_zd_nids.utils.io import (
    append_jsonl,
    ensure_parent,
    save_metadata_atomic,
    write_csv,
    write_json,
)


def test_write_json_creates_file(tmp_path):
    p = tmp_path / "out.json"
    write_json(p, {"key": "value"})
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["key"] == "value"


def test_write_json_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "out.json"
    write_json(p, {"x": 1})
    assert p.exists()


def test_append_jsonl(tmp_path):
    p = tmp_path / "events.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"b": 2})
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1
    assert json.loads(lines[1])["b"] == 2


def test_write_csv(tmp_path):
    p = tmp_path / "out.csv"
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    write_csv(p, df)
    loaded = pd.read_csv(p)
    assert list(loaded.columns) == ["x", "y"]
    assert len(loaded) == 2


def test_save_metadata_atomic(tmp_path):
    p = tmp_path / "meta.json"
    save_metadata_atomic(p, {"version": 1, "threshold": 0.5})
    data = json.loads(p.read_text())
    assert data["version"] == 1


def test_ensure_parent(tmp_path):
    p = tmp_path / "nested" / "dir" / "file.txt"
    ensure_parent(p)
    assert p.parent.exists()
