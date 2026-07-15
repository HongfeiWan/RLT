#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Run the archived holdout evaluator with strict checkpoint/cache contracts."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch

from groot_rlt.integration.artifact_lineage import canonical_json_sha256, file_sha256
from groot_rlt.representation.extract_true_prefix_holdout import HOLDOUT_CACHE_SCHEMA
from groot_rlt.representation.train_vl_embedding_autoencoder import (
    RL_TOKEN_ARCHITECTURE,
    RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
    VLTokenAutoencoder,
    VLTokenAutoencoderConfig,
    load_autoencoder_state_dict,
    validate_strict_checkpoint_payload,
)


def make_adapter_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--legacy-evaluator", required=True)
    parser.add_argument(
        "--fail-on-not-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def make_contract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-5k", required=True)
    parser.add_argument("--checkpoint-10k", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--training-cache-dir", required=True)
    parser.add_argument("--holdout-prefix-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.get("fingerprint")
    material = {key: value for key, value in manifest.items() if key != "fingerprint"}
    actual = canonical_json_sha256(material)
    _require(recorded == actual, f"Manifest fingerprint mismatch at {path}: {recorded!r} != {actual!r}")
    return manifest


def _load_legacy_evaluator(path: Path) -> ModuleType:
    _require(path.is_file(), f"Archived evaluator does not exist: {path}")
    spec = importlib.util.spec_from_file_location("groot_rlt_archived_holdout_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load archived evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_checkpoint_header(
    path: Path,
    *,
    expected_step: int,
    expected_config: dict[str, Any],
    checkpoint_fingerprint: str,
    cache_fingerprint: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    validate_strict_checkpoint_payload(payload)
    _require(
        payload.get("schema_version") == RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
        f"{path} has legacy checkpoint schema {payload.get('schema_version')!r}",
    )
    _require(
        payload.get("architecture") == RL_TOKEN_ARCHITECTURE,
        f"{path} architecture={payload.get('architecture')!r}",
    )
    _require(int(payload.get("step", -1)) == expected_step, f"{path} step is not {expected_step}")
    _require(payload.get("autoencoder_config") == expected_config, f"{path} model config drifted")
    _require("autoencoder" in payload, f"{path} has no EMA autoencoder state")
    _require("autoencoder_raw" in payload, f"{path} has no raw autoencoder state")
    lineage = payload.get("args", {}).get("representation_lineage")
    expected_lineage = {
        "cache_fingerprint": cache_fingerprint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "feature_tap": "raw_backbone_pre_action_head",
        "processor_mode": "eval",
    }
    _require(lineage == expected_lineage, f"{path} representation lineage mismatch: {lineage!r}")
    return {
        "path": str(path),
        "step": expected_step,
        "schema_version": payload["schema_version"],
        "architecture": payload["architecture"],
        "lineage": lineage,
        "last_loss": float(payload["last_loss"]),
    }


def _validate_training_cache(manifest: dict[str, Any]) -> None:
    expected = {
        "schema_version": 2,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": "raw_backbone_pre_action_head",
        "processor_mode": "eval",
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 192,
        "cache_dtype": "bfloat16",
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"Training cache {key}={manifest.get(key)!r}, expected={value!r}")


def _validate_holdout_cache(manifest: dict[str, Any], checkpoint_fingerprint: str) -> None:
    expected = {
        "schema_version": HOLDOUT_CACHE_SCHEMA,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": "raw_backbone_pre_action_head",
        "processor_mode": "eval",
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 192,
        "cache_dtype": "bfloat16",
        "transform": "eval",
        "checkpoint_fingerprint": checkpoint_fingerprint,
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"Holdout cache {key}={manifest.get(key)!r}, expected={value!r}")
    _require(manifest.get("raw_episode_id_overlap") == [], "Holdout raw episode IDs overlap training")


def _load_prefix_shards(
    directory: Path,
    manifest: dict[str, Any],
    *,
    verify_sha256: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for shard_info in manifest["shards"]:
        shard_path = directory / shard_info["file"]
        _require(shard_path.is_file(), f"Missing prefix shard: {shard_path}")
        if verify_sha256:
            actual_hash = file_sha256(shard_path)
            _require(
                actual_hash == shard_info.get("sha256"),
                f"Prefix shard hash mismatch: {shard_path}",
            )
        shard = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=True)
        packed = shard["packed"]
        mask = shard["packed_mask"].bool()
        image_mask = shard["packed_image_mask"].bool()
        _require(packed.dtype == torch.bfloat16, f"{shard_path} is not BF16")
        _require(tuple(packed.shape[1:]) == (192, 2048), f"{shard_path} shape={tuple(packed.shape)}")
        _require(bool(mask.all()) and bool(image_mask.all()), f"{shard_path} is not direct image/192")
        _require(bool(torch.isfinite(packed).all()), f"{shard_path} contains NaN/Inf")
        chunks.append(packed)
        masks.append(mask)
    prefixes = torch.cat(chunks, dim=0)
    packed_mask = torch.cat(masks, dim=0)
    _require(len(prefixes) == int(manifest["num_samples"]), "Prefix shard sample count mismatch")
    return prefixes, packed_mask


def _install_strict_loaders(
    module: ModuleType,
    *,
    training_manifest: dict[str, Any],
    holdout_manifest: dict[str, Any],
    expected_config: dict[str, Any],
    checkpoint_fingerprint: str,
    cache_fingerprint: str,
) -> None:
    def load_holdout(prefix_dir: Path):
        prefixes, mask = _load_prefix_shards(
            prefix_dir,
            holdout_manifest,
            verify_sha256=True,
        )
        _require(bool(mask.all()), "Holdout cache contains padding")
        metadata_path = prefix_dir / "metadata.csv"
        robot_path = prefix_dir / "robot_metadata.npz"
        _require(
            file_sha256(metadata_path) == holdout_manifest["metadata_csv_sha256"],
            "Holdout metadata.csv hash mismatch",
        )
        _require(
            file_sha256(robot_path) == holdout_manifest["robot_metadata_npz_sha256"],
            "Holdout robot_metadata.npz hash mismatch",
        )
        metadata = pd.read_csv(metadata_path)
        robot = np.load(robot_path)
        state = robot["state"].astype(np.float32)
        action = robot["action"].astype(np.float32)
        _require(len(metadata) == len(prefixes), "Holdout metadata/prefix count mismatch")
        _require(state.shape == (len(prefixes), 26), f"Holdout state shape={state.shape}")
        _require(action.shape == (len(prefixes), 19), f"Holdout action shape={action.shape}")
        return module.EvalSet(
            name="holdout",
            prefixes=prefixes,
            metadata=metadata,
            state=state,
            action=action,
        )

    def load_training_sample(cache_dir: Path, sample_count: int, seed: int):
        shard_count = len(training_manifest["shards"])
        _require(sample_count >= shard_count, "train-samples must be at least the cache shard count")
        base = sample_count // shard_count
        remainder = sample_count % shard_count
        rng = np.random.default_rng(seed)
        chunks: list[torch.Tensor] = []
        metadata_rows: list[dict[str, int]] = []
        global_offset = 0
        for shard_index, shard_info in enumerate(training_manifest["shards"]):
            shard_path = cache_dir / shard_info["file"]
            shard = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=True)
            count = int(shard["packed"].shape[0])
            take = min(count, base + int(shard_index < remainder))
            selected = np.sort(rng.choice(count, size=take, replace=False))
            selected_tensor = torch.from_numpy(selected)
            packed = shard["packed"][selected_tensor]
            mask = shard["packed_mask"][selected_tensor].bool()
            image_mask = shard["packed_image_mask"][selected_tensor].bool()
            _require(packed.dtype == torch.bfloat16, f"{shard_path} is not BF16")
            _require(tuple(packed.shape[1:]) == (192, 2048), f"{shard_path} shape drifted")
            _require(bool(mask.all()) and bool(image_mask.all()), f"{shard_path} is not image/192")
            _require(bool(torch.isfinite(packed).all()), f"{shard_path} selected rows contain NaN/Inf")
            chunks.append(packed)
            metadata_rows.extend(
                {
                    "sample_index": global_offset + row_in_shard,
                    "shard_index": shard_index,
                    "row_in_shard": row_in_shard,
                }
                for row_in_shard in selected.tolist()
            )
            global_offset += count
        prefixes = torch.cat(chunks, dim=0)
        _require(len(prefixes) == sample_count, f"Loaded {len(prefixes)} training rows, expected {sample_count}")
        return module.EvalSet(
            name="train",
            prefixes=prefixes,
            metadata=pd.DataFrame(metadata_rows),
        )

    def load_model(
        config: VLTokenAutoencoderConfig,
        device: torch.device,
        checkpoint: Path | None,
        weight_key: str | None,
        seed: int,
    ) -> VLTokenAutoencoder:
        _require(asdict(config) == expected_config, "Evaluator model config differs from training config")
        torch.manual_seed(seed)
        model = VLTokenAutoencoder(config)
        if checkpoint is not None and weight_key is not None:
            payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            validate_strict_checkpoint_payload(payload)
            lineage = payload.get("args", {}).get("representation_lineage")
            _require(
                lineage
                == {
                    "cache_fingerprint": cache_fingerprint,
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                    "feature_tap": "raw_backbone_pre_action_head",
                    "processor_mode": "eval",
                },
                f"Checkpoint lineage drifted while loading {checkpoint}",
            )
            _require(weight_key in payload, f"{weight_key!r} not found in {checkpoint}")
            load_autoencoder_state_dict(model, payload[weight_key])
            del payload
            gc.collect()
        model.to(device=device, dtype=torch.float32)
        model.eval()
        return model

    module.load_holdout = load_holdout
    module.load_training_sample = load_training_sample
    module.load_model = load_model


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    adapter_args, forwarded = make_adapter_parser().parse_known_args()
    contract_args, _ = make_contract_parser().parse_known_args(forwarded)

    legacy_path = Path(adapter_args.legacy_evaluator).expanduser().resolve()
    checkpoint_5k = Path(contract_args.checkpoint_5k).expanduser().resolve()
    checkpoint_10k = Path(contract_args.checkpoint_10k).expanduser().resolve()
    training_config_path = Path(contract_args.training_config).expanduser().resolve()
    training_cache_dir = Path(contract_args.training_cache_dir).expanduser().resolve()
    holdout_prefix_dir = Path(contract_args.holdout_prefix_dir).expanduser().resolve()
    output_dir = Path(contract_args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to mix holdout evaluation artifacts: {output_dir}")

    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    _require(training_config.get("schema_version") == 2, "Training config schema is not strict v2")
    _require(training_config.get("architecture") == RL_TOKEN_ARCHITECTURE, "Training architecture drifted")
    expected_config = dict(training_config["autoencoder_config"])

    training_manifest = _load_manifest(training_cache_dir / "manifest.json")
    _validate_training_cache(training_manifest)
    checkpoint_fingerprint = str(training_manifest["checkpoint_fingerprint"])
    cache_fingerprint = str(training_manifest["fingerprint"])
    holdout_manifest = _load_manifest(holdout_prefix_dir / "manifest.json")
    _validate_holdout_cache(holdout_manifest, checkpoint_fingerprint)

    checkpoint_contracts = [
        _validate_checkpoint_header(
            checkpoint_5k,
            expected_step=5_000,
            expected_config=expected_config,
            checkpoint_fingerprint=checkpoint_fingerprint,
            cache_fingerprint=cache_fingerprint,
        ),
        _validate_checkpoint_header(
            checkpoint_10k,
            expected_step=10_000,
            expected_config=expected_config,
            checkpoint_fingerprint=checkpoint_fingerprint,
            cache_fingerprint=cache_fingerprint,
        ),
    ]
    module = _load_legacy_evaluator(legacy_path)
    _install_strict_loaders(
        module,
        training_manifest=training_manifest,
        holdout_manifest=holdout_manifest,
        expected_config=expected_config,
        checkpoint_fingerprint=checkpoint_fingerprint,
        cache_fingerprint=cache_fingerprint,
    )

    original_argv = sys.argv
    try:
        sys.argv = [str(legacy_path), *forwarded]
        module.main()
    finally:
        sys.argv = original_argv

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact_contract"] = {
        "adapter": "groot_rlt.representation.evaluate_true_prefix_holdout",
        "legacy_evaluator": str(legacy_path),
        "legacy_evaluator_sha256": file_sha256(legacy_path),
        "training_config": str(training_config_path),
        "training_config_sha256": file_sha256(training_config_path),
        "groot_checkpoint_fingerprint": checkpoint_fingerprint,
        "training_cache_fingerprint": cache_fingerprint,
        "holdout_cache_fingerprint": holdout_manifest["fingerprint"],
        "holdout_raw_episode_id_overlap": holdout_manifest["raw_episode_id_overlap"],
        "prefix_dtype": "bfloat16",
        "prefix_contract": "direct image/uniform/192; no secondary token selection",
        "checkpoints": checkpoint_contracts,
    }
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary["artifact_contract"], indent=2, ensure_ascii=False))
    if adapter_args.fail_on_not_ready and summary.get("verdict") != "trained_and_functional":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
