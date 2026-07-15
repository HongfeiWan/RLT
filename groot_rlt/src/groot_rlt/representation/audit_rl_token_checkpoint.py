#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed audit for a strict RL-token checkpoint and its prefix cache."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from groot_rlt.integration.artifact_lineage import (
    canonical_json_sha256,
    checkpoint_fingerprint,
    file_sha256,
)

EXPECTED_CHECKPOINT_SCHEMA_VERSION = 2
EXPECTED_ARCHITECTURE = "openpi_rlt_strict_cross_attention_v1"
EXPECTED_CACHE_SCHEMA_VERSION = 2
EXPECTED_FEATURE_TAP = "raw_backbone_pre_action_head"
EXPECTED_PROCESSOR_MODE = "eval"
STRICT_STATE_SENTINELS = {
    "query_token",
    "encoder_memory_pos",
    "decoder_query",
    "decoder_memory_pos",
}


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-step", type=int, default=10_000)
    parser.add_argument("--expected-token-scope", default="image")
    parser.add_argument("--expected-token-sampling", default="uniform")
    parser.add_argument("--expected-max-vl-tokens", type=int, default=192)
    parser.add_argument("--expected-model-dim", type=int, default=2048)
    parser.add_argument("--expected-cache-dtype", default="bfloat16")
    parser.add_argument(
        "--verify-optimizer-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan all floating optimizer tensors for NaN/Inf.",
    )
    parser.add_argument(
        "--verify-cache-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-hash every cache shard rather than trusting the manifest.",
    )
    parser.add_argument("--output-json", required=True)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _iter_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "<root>", value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_tensors(item, child)


def _tensor_tree_summary(value: Any, *, label: str) -> dict[str, Any]:
    tensor_count = 0
    floating_tensor_count = 0
    numel = 0
    floating_numel = 0
    dtypes: dict[str, int] = {}
    for name, tensor in _iter_tensors(value):
        tensor_count += 1
        tensor_numel = int(tensor.numel())
        numel += tensor_numel
        dtype_name = str(tensor.dtype)
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + tensor_numel
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            floating_tensor_count += 1
            floating_numel += tensor_numel
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{label}.{name} contains NaN or Inf")
    _require(tensor_count > 0, f"{label} contains no tensors")
    return {
        "tensor_count": tensor_count,
        "floating_tensor_count": floating_tensor_count,
        "numel": numel,
        "floating_numel": floating_numel,
        "dtype_numel": dict(sorted(dtypes.items())),
        "all_finite": True,
    }


def _validate_state_contract(
    state: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    missing = sorted(STRICT_STATE_SENTINELS - set(state))
    _require(not missing, f"{label} is missing strict state keys: {missing}")

    model_dim = int(config["model_dim"])
    max_tokens = int(config["max_vl_tokens"])
    expected_shapes = {
        "query_token": (1, model_dim),
        "encoder_memory_pos": (max_tokens, model_dim),
        "decoder_query": (max_tokens, model_dim),
        "decoder_memory_pos": (1, model_dim),
    }
    for name, shape in expected_shapes.items():
        actual = tuple(state[name].shape)
        _require(actual == shape, f"{label}.{name} shape={actual}, expected={shape}")
    return _tensor_tree_summary(state, label=label)


def _validate_cache_shard(
    cache_dir: Path,
    shard_info: Mapping[str, Any],
    *,
    expected_max_tokens: int,
    expected_model_dim: int,
) -> dict[str, Any]:
    path = cache_dir / str(shard_info["file"])
    _require(path.is_file(), f"Missing cache shard: {path}")
    shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    required = {
        "packed",
        "packed_mask",
        "packed_image_mask",
        "token_counts",
        "selected_counts",
    }
    _require(required <= set(shard), f"Cache shard {path} is missing {sorted(required - set(shard))}")
    packed = shard["packed"]
    packed_mask = shard["packed_mask"].bool()
    packed_image_mask = shard["packed_image_mask"].bool()
    expected_shape = (int(shard_info["num_samples"]), expected_max_tokens, expected_model_dim)
    _require(tuple(packed.shape) == expected_shape, f"{path} packed={tuple(packed.shape)}, expected={expected_shape}")
    _require(packed.dtype == torch.bfloat16, f"{path} packed dtype={packed.dtype}, expected=torch.bfloat16")
    _require(tuple(packed_mask.shape) == expected_shape[:2], f"{path} packed_mask shape mismatch")
    _require(tuple(packed_image_mask.shape) == expected_shape[:2], f"{path} image_mask shape mismatch")
    _require(bool(packed_mask.all()), f"{path} contains padding in an image/192 cache")
    _require(bool(packed_image_mask.all()), f"{path} contains non-image selected tokens")
    _require(bool(torch.isfinite(packed).all()), f"{path} contains NaN or Inf")
    token_counts = {int(value) for value in shard["token_counts"]}
    selected_counts = {int(value) for value in shard["selected_counts"]}
    _require(token_counts == {expected_max_tokens}, f"{path} token_counts={sorted(token_counts)}")
    _require(
        selected_counts == {expected_max_tokens},
        f"{path} selected_counts={sorted(selected_counts)}",
    )
    return {
        "file": str(path),
        "shape": list(packed.shape),
        "dtype": str(packed.dtype),
        "all_finite": True,
        "all_valid_image_tokens": True,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    cache_manifest_path = Path(args.cache_manifest).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    _require(checkpoint_path.is_file(), f"Missing RL-token checkpoint: {checkpoint_path}")
    _require(cache_manifest_path.is_file(), f"Missing cache manifest: {cache_manifest_path}")
    _require(model_path.is_dir(), f"Missing GR00T checkpoint directory: {model_path}")

    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    recorded_cache_fingerprint = cache_manifest.get("fingerprint")
    cache_material = {key: value for key, value in cache_manifest.items() if key != "fingerprint"}
    actual_cache_fingerprint = canonical_json_sha256(cache_material)
    _require(
        recorded_cache_fingerprint == actual_cache_fingerprint,
        "Cache manifest fingerprint mismatch: "
        f"recorded={recorded_cache_fingerprint!r} actual={actual_cache_fingerprint!r}",
    )
    expected_cache_fields = {
        "schema_version": EXPECTED_CACHE_SCHEMA_VERSION,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": EXPECTED_FEATURE_TAP,
        "processor_mode": EXPECTED_PROCESSOR_MODE,
        "token_scope": args.expected_token_scope,
        "token_sampling": args.expected_token_sampling,
        "max_vl_tokens": int(args.expected_max_vl_tokens),
        "cache_dtype": args.expected_cache_dtype,
    }
    for key, expected in expected_cache_fields.items():
        _require(
            cache_manifest.get(key) == expected,
            f"Cache manifest {key}={cache_manifest.get(key)!r}, expected={expected!r}",
        )

    actual_model_fingerprint, model_file_hashes = checkpoint_fingerprint(model_path)
    _require(
        cache_manifest.get("checkpoint_fingerprint") == actual_model_fingerprint,
        "Cache is not bound to the requested GR00T checkpoint: "
        f"cache={cache_manifest.get('checkpoint_fingerprint')!r} "
        f"actual={actual_model_fingerprint!r}",
    )
    _require(
        cache_manifest.get("checkpoint_files") == model_file_hashes,
        "Cache checkpoint component hashes differ from the requested GR00T checkpoint",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    _require(isinstance(checkpoint, dict), "RL-token checkpoint payload must be a mapping")
    _require(
        checkpoint.get("schema_version") == EXPECTED_CHECKPOINT_SCHEMA_VERSION,
        f"Checkpoint schema={checkpoint.get('schema_version')!r}, expected=2",
    )
    _require(
        checkpoint.get("architecture") == EXPECTED_ARCHITECTURE,
        f"Checkpoint architecture={checkpoint.get('architecture')!r}, expected={EXPECTED_ARCHITECTURE!r}",
    )
    _require(
        int(checkpoint.get("step", -1)) == int(args.expected_step),
        f"Checkpoint step={checkpoint.get('step')!r}, expected={args.expected_step}",
    )
    last_loss = float(checkpoint.get("last_loss", math.nan))
    _require(math.isfinite(last_loss), f"Checkpoint last_loss is not finite: {last_loss}")
    ema_decay = float(checkpoint.get("ema_decay", math.nan))
    _require(math.isclose(ema_decay, 0.99), f"Checkpoint ema_decay={ema_decay}, expected=0.99")

    config = checkpoint.get("autoencoder_config")
    _require(isinstance(config, dict), "Checkpoint has no autoencoder_config mapping")
    expected_config = {
        "input_dim": int(args.expected_model_dim),
        "model_dim": int(args.expected_model_dim),
        "rl_token_dim": int(args.expected_model_dim),
        "max_vl_tokens": int(args.expected_max_vl_tokens),
        "encoder_layers": 2,
        "decoder_layers": 2,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "use_prefix_mask_token": False,
        "use_decoder_cross_attention": True,
    }
    for key, expected in expected_config.items():
        _require(
            config.get(key) == expected,
            f"autoencoder_config.{key}={config.get(key)!r}, expected={expected!r}",
        )

    checkpoint_args = checkpoint.get("args")
    _require(isinstance(checkpoint_args, dict), "Checkpoint has no args mapping")
    expected_training_args = {
        "token_scope": args.expected_token_scope,
        "token_sampling": args.expected_token_sampling,
        "max_vl_tokens": int(args.expected_max_vl_tokens),
        "cache_dtype": args.expected_cache_dtype,
        "decoder_cross_attention": True,
        "decoder_prefix_corruption": False,
        "autoencoder_bf16": False,
    }
    for key, expected in expected_training_args.items():
        _require(
            checkpoint_args.get(key) == expected,
            f"checkpoint args {key}={checkpoint_args.get(key)!r}, expected={expected!r}",
        )
    lineage = checkpoint_args.get("representation_lineage")
    _require(isinstance(lineage, dict), "Checkpoint has no representation_lineage")
    expected_lineage = {
        "cache_fingerprint": actual_cache_fingerprint,
        "checkpoint_fingerprint": actual_model_fingerprint,
        "feature_tap": EXPECTED_FEATURE_TAP,
        "processor_mode": EXPECTED_PROCESSOR_MODE,
    }
    _require(lineage == expected_lineage, f"Checkpoint lineage mismatch: {lineage!r}")

    ema_state = checkpoint.get("autoencoder")
    raw_state = checkpoint.get("autoencoder_raw")
    _require(isinstance(ema_state, dict), "Checkpoint is missing EMA autoencoder state")
    _require(isinstance(raw_state, dict), "Checkpoint is missing raw autoencoder state")
    _require(set(ema_state) == set(raw_state), "EMA and raw autoencoder state keys differ")
    for name in ema_state:
        _require(
            tuple(ema_state[name].shape) == tuple(raw_state[name].shape),
            f"EMA/raw shape mismatch for {name}",
        )
    ema_summary = _validate_state_contract(ema_state, config=config, label="autoencoder_ema")
    raw_summary = _validate_state_contract(raw_state, config=config, label="autoencoder_raw")
    if int(args.expected_model_dim) == 2048 and int(args.expected_max_vl_tokens) == 192:
        for label, summary in (("EMA", ema_summary), ("raw", raw_summary)):
            _require(
                summary["tensor_count"] == 116,
                f"{label} state has {summary['tensor_count']} tensors, expected 116",
            )
            _require(
                summary["numel"] == 806_318_080,
                f"{label} state has {summary['numel']} values, expected 806318080",
            )
            _require(
                summary["dtype_numel"] == {"torch.float32": 806_318_080},
                f"{label} state is not entirely FP32: {summary['dtype_numel']}",
            )
    optimizer_summary: dict[str, Any] | None = None
    if args.verify_optimizer_finite:
        optimizer_summary = _tensor_tree_summary(
            checkpoint.get("optimizer"),
            label="optimizer",
        )

    shards = cache_manifest.get("shards")
    _require(isinstance(shards, list) and shards, "Cache manifest contains no shards")
    cache_dir = cache_manifest_path.parent
    cache_hashes: dict[str, str] = {}
    if args.verify_cache_sha256:
        for shard in shards:
            shard_path = cache_dir / str(shard["file"])
            _require(shard_path.is_file(), f"Missing cache shard: {shard_path}")
            actual_hash = file_sha256(shard_path)
            _require(
                shard.get("sha256") == actual_hash,
                f"Cache shard hash mismatch for {shard_path}: "
                f"recorded={shard.get('sha256')!r} actual={actual_hash!r}",
            )
            cache_hashes[str(shard["file"])] = actual_hash
    boundary_shards = [
        _validate_cache_shard(
            cache_dir,
            shard,
            expected_max_tokens=int(args.expected_max_vl_tokens),
            expected_model_dim=int(args.expected_model_dim),
        )
        for shard in (shards[0], shards[-1])
    ]

    checkpoint_hash = file_sha256(checkpoint_path)
    return {
        "verdict": "pass",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint_path.stat().st_size,
            "schema_version": checkpoint["schema_version"],
            "architecture": checkpoint["architecture"],
            "step": int(checkpoint["step"]),
            "last_loss": last_loss,
            "ema_decay": ema_decay,
            "autoencoder_config": config,
            "ema": ema_summary,
            "raw": raw_summary,
            "optimizer": optimizer_summary,
            "lineage": lineage,
        },
        "groot_checkpoint": {
            "path": str(model_path),
            "fingerprint": actual_model_fingerprint,
            "component_sha256": model_file_hashes,
        },
        "prefix_cache": {
            "manifest": str(cache_manifest_path),
            "fingerprint": actual_cache_fingerprint,
            "num_samples": int(cache_manifest["num_samples"]),
            "num_valid_tokens": int(cache_manifest["num_valid_tokens"]),
            "num_shards": len(shards),
            "all_shard_sha256_verified": bool(args.verify_cache_sha256),
            "shard_sha256": cache_hashes,
            "boundary_shards": boundary_shards,
        },
    }


def main() -> None:
    args = make_arg_parser().parse_args()
    result = audit(args)
    output_json = Path(args.output_json).expanduser().resolve()
    _write_json_atomic(output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
