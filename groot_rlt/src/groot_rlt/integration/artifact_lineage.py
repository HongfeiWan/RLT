#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Content fingerprints for GR00T and RLT artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_hashes(model_path: str | Path) -> dict[str, str]:
    """Hash the complete inference contract saved with a GR00T checkpoint."""

    root = Path(model_path).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint index has no weight_map: {index_path}")

    required = {
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        *weight_map.values(),
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete at {root}: missing={missing}")
    return {name: file_sha256(root / name) for name in sorted(required)}


def checkpoint_fingerprint(model_path: str | Path) -> tuple[str, dict[str, str]]:
    """Return the semantic fingerprint and component hashes for a checkpoint."""

    files = checkpoint_file_hashes(model_path)
    return canonical_json_sha256(files), files
