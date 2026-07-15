# SPDX-License-Identifier: Apache-2.0

"""Fail-closed deployment contract for a schema-2 VL-prefix cache."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groot_rlt.integration.artifact_lineage import canonical_json_sha256, file_sha256

FEATURE_TAP = "raw_backbone_pre_action_head"
PROCESSOR_MODE = "eval"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def require_sha256(value: Any, label: str) -> str:
    """Return a validated lower-case SHA-256 fingerprint."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must have form sha256:<64 lowercase hex>, got {value!r}")
    return value


def _absolute_recorded_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute, got {value!r}")
    return str(path.resolve())


@dataclass(frozen=True)
class PrefixCacheContract:
    """Serving-relevant contract recovered from a signed cache manifest."""

    fingerprint: str
    checkpoint_fingerprint: str
    token_scope: str
    token_sampling: str
    max_vl_tokens: int
    input_dim: int
    video_modality_keys: tuple[str, ...]
    feature_tap: str
    processor_mode: str
    model_path: str
    processor_path: str
    vlm_model_path: str
    manifest_path: str


def load_prefix_cache_contract(
    manifest_path: str | Path,
    *,
    expected_cache_fingerprint: str | None = None,
    expected_checkpoint_fingerprint: str | None = None,
) -> PrefixCacheContract:
    """Validate a cache manifest and recover its immutable deployment paths."""

    if expected_cache_fingerprint is not None:
        expected_cache_fingerprint = require_sha256(
            expected_cache_fingerprint, "expected_cache_fingerprint"
        )
    if expected_checkpoint_fingerprint is not None:
        expected_checkpoint_fingerprint = require_sha256(
            expected_checkpoint_fingerprint, "expected_checkpoint_fingerprint"
        )
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prefix-cache manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Prefix-cache manifest must be a JSON object")
    recorded_fingerprint = require_sha256(
        manifest.get("fingerprint"), "prefix_cache_manifest.fingerprint"
    )
    material = {key: value for key, value in manifest.items() if key != "fingerprint"}
    actual_fingerprint = canonical_json_sha256(material)
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError(
            "Prefix-cache manifest fingerprint is invalid: "
            f"recorded={recorded_fingerprint} actual={actual_fingerprint}"
        )
    if (
        expected_cache_fingerprint is not None
        and actual_fingerprint != expected_cache_fingerprint
    ):
        raise ValueError(
            "Prefix-cache manifest does not match expected cache fingerprint: "
            f"manifest={actual_fingerprint} expected={expected_cache_fingerprint}"
        )
    if manifest.get("schema_version") != 2:
        raise ValueError(
            f"Prefix-cache schema_version={manifest.get('schema_version')!r}, expected 2"
        )
    if manifest.get("representation_source") != "groot_checkpoint_backbone":
        raise ValueError("Prefix cache was not extracted from the GR00T checkpoint backbone")
    if manifest.get("feature_tap") != FEATURE_TAP:
        raise ValueError(
            f"Prefix-cache feature_tap={manifest.get('feature_tap')!r}, expected {FEATURE_TAP!r}"
        )
    if manifest.get("processor_mode") != PROCESSOR_MODE:
        raise ValueError(
            "Prefix-cache processor_mode="
            f"{manifest.get('processor_mode')!r}, expected {PROCESSOR_MODE!r}"
        )
    checkpoint_value = require_sha256(
        manifest.get("checkpoint_fingerprint"),
        "prefix_cache_manifest.checkpoint_fingerprint",
    )
    if (
        expected_checkpoint_fingerprint is not None
        and checkpoint_value != expected_checkpoint_fingerprint
    ):
        raise ValueError(
            "Prefix-cache 400k fingerprint mismatch: "
            f"manifest={checkpoint_value} expected={expected_checkpoint_fingerprint}"
        )
    token_scope = manifest.get("token_scope")
    if token_scope not in {"all", "image", "non_image"}:
        raise ValueError(f"Invalid prefix-cache token_scope={token_scope!r}")
    token_sampling = manifest.get("token_sampling")
    if token_sampling not in {"head", "tail", "uniform"}:
        raise ValueError(
            "Prefix-cache token_sampling must be deterministic head/tail/uniform, "
            f"got {token_sampling!r}"
        )
    max_vl_tokens = manifest.get("max_vl_tokens")
    input_dim = manifest.get("input_dim")
    if type(max_vl_tokens) is not int or max_vl_tokens < 1:
        raise ValueError("Prefix-cache max_vl_tokens must be a positive integer")
    if type(input_dim) is not int or input_dim < 1:
        raise ValueError("Prefix-cache input_dim must be a positive integer")
    video_keys = manifest.get("video_modality_keys")
    if (
        not isinstance(video_keys, list)
        or not video_keys
        or not all(isinstance(key, str) and key for key in video_keys)
        or len(set(video_keys)) != len(video_keys)
    ):
        raise ValueError("Prefix-cache video_modality_keys must be unique non-empty strings")
    return PrefixCacheContract(
        fingerprint=actual_fingerprint,
        checkpoint_fingerprint=checkpoint_value,
        token_scope=token_scope,
        token_sampling=token_sampling,
        max_vl_tokens=max_vl_tokens,
        input_dim=input_dim,
        video_modality_keys=tuple(video_keys),
        feature_tap=FEATURE_TAP,
        processor_mode=PROCESSOR_MODE,
        model_path=_absolute_recorded_path(
            manifest.get("base_model_path"), "prefix_cache_manifest.base_model_path"
        ),
        processor_path=_absolute_recorded_path(
            manifest.get("processor_path"), "prefix_cache_manifest.processor_path"
        ),
        vlm_model_path=_absolute_recorded_path(
            manifest.get("vlm_model_path"), "prefix_cache_manifest.vlm_model_path"
        ),
        manifest_path=str(path),
    )


def validate_prefix_cache_deployment_paths(
    contract: PrefixCacheContract,
    *,
    model_path: str | Path,
    processor_path: str | Path,
    vlm_model_path: str | Path,
    context: str,
) -> tuple[Path, Path, Path]:
    """Require all runtime model inputs to equal the signed cache paths."""

    resolved = (
        Path(model_path).expanduser().resolve(),
        Path(processor_path).expanduser().resolve(),
        Path(vlm_model_path).expanduser().resolve(),
    )
    for name, actual, recorded in zip(
        ("model_path", "processor_path", "vlm_model_path"),
        resolved,
        (contract.model_path, contract.processor_path, contract.vlm_model_path),
        strict=True,
    ):
        if str(actual) != recorded:
            raise ValueError(
                f"{context} {name}={actual} differs from signed prefix-cache path {recorded}"
            )
    return resolved


def vlm_content_fingerprint(vlm_model_path: str | Path) -> str:
    """Hash every regular file in a local VLM deployment directory.

    This is a deployment-time integrity check. It is deliberately not presented
    as part of the historical representation-training lineage, because existing
    cache manifests did not record VLM file hashes.
    """

    root = Path(vlm_model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VLM model directory does not exist: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    if not files:
        raise ValueError(f"VLM model directory contains no files: {root}")
    hashes = {path.relative_to(root).as_posix(): file_sha256(path) for path in files}
    return canonical_json_sha256(hashes)
