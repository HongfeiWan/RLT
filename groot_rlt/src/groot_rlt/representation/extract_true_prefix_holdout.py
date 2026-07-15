#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Extract a lineage-bound 400k VLA-prefix cache from unseen LeRobot episodes."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from groot_rlt.integration.artifact_lineage import (
    canonical_json_sha256,
    checkpoint_fingerprint,
    file_sha256,
)
from groot_rlt.integration.checkpoint_policy_utils import resolve_processor_path
from groot_rlt.representation.train_vl_embedding_autoencoder import (
    VL_CACHE_FEATURE_TAP,
    VL_CACHE_PROCESSOR_MODE,
    autocast_context,
    build_backbone,
    build_dataset_and_processor,
    move_backbone_inputs,
    pack_vl_tokens,
    seed_everything,
)

HOLDOUT_CACHE_SCHEMA = "groot_rlt.true_prefix_holdout.v1"


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--training-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modality-config-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--vlm-model-path", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_episode_ids(dataset_dir: Path) -> set[str]:
    rows = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    return {
        str(row.get("teleop_stack_metadata", {}).get("raw_episode_id"))
        for row in rows
        if row.get("teleop_stack_metadata", {}).get("raw_episode_id")
    }


def normalize_instruction(text: str, formalize: bool) -> str:
    if not formalize:
        return text
    return re.sub(r"[^\w\s]", "", text.lower())


def _dataset_metadata_hashes(dataset_dir: Path) -> dict[str, str]:
    meta_dir = dataset_dir / "meta"
    names = ("info.json", "episodes.jsonl", "modality.json", "tasks.jsonl")
    missing = [name for name in names if not (meta_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset metadata is incomplete at {meta_dir}: missing={missing}")
    return {name: file_sha256(meta_dir / name) for name in names}


def _row_array(row: Any, key: str, expected_dim: int) -> np.ndarray:
    values = np.asarray(row[key], dtype=np.float32)
    if values.shape != (expected_dim,):
        raise ValueError(f"{key} shape={values.shape}, expected={(expected_dim,)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains NaN or Inf")
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = make_arg_parser().parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")
    if args.shard_size % args.batch_size != 0:
        raise ValueError("shard-size must be divisible by batch-size for deterministic shards")
    if args.frame_stride < 1:
        raise ValueError("frame-stride must be positive")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to mix holdout artifacts into existing path: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    working_dir = output_dir.with_name(f".{output_dir.name}.partial.{os.getpid()}")
    if working_dir.exists():
        raise FileExistsError(f"Stale partial holdout directory exists: {working_dir}")
    working_dir.mkdir()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    training_dataset_dir = Path(args.training_dataset_dir).expanduser().resolve()
    base_model_path = Path(args.base_model_path).expanduser().resolve()
    processor_path = (
        Path(args.processor_path).expanduser().resolve()
        if args.processor_path is not None
        else resolve_processor_path(base_model_path)
    )
    for label, path in (
        ("holdout dataset", dataset_dir),
        ("training dataset", training_dataset_dir),
        ("400k checkpoint", base_model_path),
        ("400k processor", processor_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    holdout_ids = raw_episode_ids(dataset_dir)
    training_ids = raw_episode_ids(training_dataset_dir)
    overlap = sorted(holdout_ids & training_ids)
    if overlap:
        raise RuntimeError(f"Holdout leakage: {len(overlap)} raw episode IDs overlap training data")
    if not holdout_ids or not training_ids:
        raise RuntimeError(
            "Cannot prove holdout disjointness because one dataset has no raw_episode_id metadata"
        )

    model_fingerprint, model_file_hashes = checkpoint_fingerprint(base_model_path)
    dataset_metadata_sha256 = _dataset_metadata_hashes(dataset_dir)
    training_dataset_metadata_sha256 = _dataset_metadata_hashes(training_dataset_dir)
    model_args = SimpleNamespace(
        base_model_path=str(base_model_path),
        processor_path=str(processor_path),
        vlm_model_path=args.vlm_model_path,
        trust_remote_code=True,
        local_files_only=True,
        load_bf16=True,
        use_flash_attention=None,
        modality_config_path=args.modality_config_path,
        embodiment_tag=args.embodiment_tag,
        dataset_dir=str(dataset_dir),
        video_backend=args.video_backend,
        episode_sampling_rate=1.0,
        seed=args.seed,
        instruction=args.instruction,
    )
    device = torch.device(args.device)
    backbone, model_cfg, loading_kwargs = build_backbone(model_args, device)
    dataset, processor = build_dataset_and_processor(model_args, model_cfg, loading_kwargs)
    processor.eval()

    episode_rows = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    loader = dataset.loader
    if len(episode_rows) != len(loader):
        raise RuntimeError(
            f"Episode metadata/loader mismatch: metadata={len(episode_rows)} loader={len(loader)}"
        )
    episode_meta = {int(row["episode_index"]): row for row in episode_rows}
    dataset_info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    image_keys = list(dataset.video_modality.modality_keys)
    language_key = dataset.language_modality.modality_keys[0]
    language_column = f"language.{language_key}"

    storage_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.cache_dtype]
    packed_buffer: list[torch.Tensor] = []
    mask_buffer: list[torch.Tensor] = []
    image_mask_buffer: list[torch.Tensor] = []
    token_count_buffer: list[int] = []
    selected_count_buffer: list[int] = []
    pending_features: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    shards: list[dict[str, Any]] = []
    buffered = 0
    extracted = 0
    started = time.time()

    def flush_shard() -> None:
        nonlocal packed_buffer, mask_buffer, image_mask_buffer
        nonlocal token_count_buffer, selected_count_buffer, buffered
        if not packed_buffer:
            return
        packed = torch.cat(packed_buffer, dim=0).to(dtype=storage_dtype, device="cpu")
        mask = torch.cat(mask_buffer, dim=0).to(dtype=torch.bool, device="cpu")
        image_mask = torch.cat(image_mask_buffer, dim=0).to(dtype=torch.bool, device="cpu")
        if packed.shape[1:] != (192, 2048):
            raise RuntimeError(f"Unexpected holdout shard shape: {tuple(packed.shape)}")
        if not bool(torch.isfinite(packed).all()):
            raise RuntimeError("Holdout shard contains NaN or Inf")
        if not bool(mask.all()) or not bool(image_mask.all()):
            raise RuntimeError("Holdout shard contains padding or non-image tokens")
        filename = f"prefix_shard_{len(shards):04d}.pt"
        shard_path = working_dir / filename
        torch.save(
            {
                "packed": packed,
                "packed_mask": mask,
                "packed_image_mask": image_mask,
                "token_counts": torch.tensor(token_count_buffer, dtype=torch.int32),
                "selected_counts": torch.tensor(selected_count_buffer, dtype=torch.int32),
            },
            shard_path,
        )
        shards.append(
            {
                "file": filename,
                "sha256": file_sha256(shard_path),
                "num_samples": int(packed.shape[0]),
                "num_valid_tokens": int(mask.sum()),
                "start_index": int(extracted - packed.shape[0]),
            }
        )
        packed_buffer = []
        mask_buffer = []
        image_mask_buffer = []
        token_count_buffer = []
        selected_count_buffer = []
        buffered = 0

    def flush_backbone() -> None:
        nonlocal pending_features, pending_rows, buffered, extracted
        if not pending_features:
            return
        batch = processor.collator(pending_features)
        dtype = next(backbone.parameters()).dtype
        backbone_inputs = move_backbone_inputs(batch["inputs"], device, dtype)
        with torch.inference_mode(), autocast_context(device, dtype == torch.bfloat16):
            backbone_output = backbone(backbone_inputs)
            packed, mask, image_mask, token_counts, selected_counts = pack_vl_tokens(
                backbone_output,
                token_scope="image",
                max_tokens=192,
                token_sampling="uniform",
            )
        if packed.shape[1:] != (192, 2048):
            raise RuntimeError(f"Unexpected packed prefix shape: {tuple(packed.shape)}")
        if not bool(mask.all()) or not bool(image_mask.all()):
            raise RuntimeError("Holdout extraction produced padding or non-image tokens")
        if set(token_counts) != {192} or set(selected_counts) != {192}:
            raise RuntimeError(
                f"Unexpected image token counts: raw={token_counts}, selected={selected_counts}"
            )

        packed_buffer.append(packed.detach().cpu())
        mask_buffer.append(mask.detach().cpu())
        image_mask_buffer.append(image_mask.detach().cpu())
        token_count_buffer.extend(int(value) for value in token_counts)
        selected_count_buffer.extend(int(value) for value in selected_counts)
        buffered += int(packed.shape[0])
        for pending in pending_rows:
            metadata_rows.append(pending["metadata"])
            states.append(pending["state"])
            actions.append(pending["action"])
        extracted += int(packed.shape[0])
        pending_features = []
        pending_rows = []
        if buffered >= args.shard_size:
            if buffered != args.shard_size:
                raise RuntimeError(
                    f"Buffered {buffered} samples, expected exact shard size {args.shard_size}"
                )
            flush_shard()
        print(
            f"extracted={extracted} shards={len(shards)} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    stop = False
    for loader_episode_index in range(len(loader)):
        if stop:
            break
        episode = loader[loader_episode_index]
        actual_episode_index = int(episode_rows[loader_episode_index]["episode_index"])
        meta = episode_meta[actual_episode_index]
        teleop = meta.get("teleop_stack_metadata", {})
        episode_length = len(episode)
        parquet_path = dataset_dir / dataset_info["data_path"].format(
            episode_chunk=actual_episode_index // int(dataset_info["chunks_size"]),
            episode_index=actual_episode_index,
        )
        raw_episode = pd.read_parquet(parquet_path)
        if len(raw_episode) != episode_length:
            raise RuntimeError(
                f"Episode length mismatch for {actual_episode_index}: "
                f"loader={episode_length}, parquet={len(raw_episode)}"
            )
        for step_index in range(0, episode_length, args.frame_stride):
            if args.max_samples is not None and extracted + len(pending_features) >= args.max_samples:
                stop = True
                break
            row = episode.iloc[step_index]
            raw_row = raw_episode.iloc[step_index]
            images = {key: [row[f"video.{key}"]] for key in image_keys}
            language = normalize_instruction(
                args.instruction or str(row[language_column]),
                processor.formalize_language,
            )
            pending_features.append(
                processor._get_vlm_inputs(
                    image_keys=image_keys,
                    images=images,
                    masks=None,
                    image_transform=processor.eval_image_transform,
                    language=language,
                )
            )
            frame_index = int(raw_row["frame_index"])
            validity = int(raw_row.get("annotation.human.validity", 1))
            pending_rows.append(
                {
                    "metadata": {
                        "sample_index": extracted + len(pending_features) - 1,
                        "loader_episode_index": loader_episode_index,
                        "episode_index": actual_episode_index,
                        "frame_index": frame_index,
                        "episode_length": episode_length,
                        "timestamp": float(raw_row["timestamp"]),
                        "progress": frame_index / max(episode_length - 1, 1),
                        "validity": validity,
                        "success": bool(teleop.get("success", meta.get("success", True))),
                        "outcome": str(teleop.get("outcome", "")),
                        "source_dataset": str(teleop.get("source_dataset", "")),
                        "raw_episode_id": str(teleop.get("raw_episode_id", "")),
                    },
                    "state": _row_array(raw_row, "observation.state", 26),
                    "action": _row_array(raw_row, "action", 19),
                }
            )
            if len(pending_features) >= args.batch_size:
                flush_backbone()
    flush_backbone()
    flush_shard()

    if not metadata_rows:
        raise RuntimeError("No holdout samples were extracted")
    state_array = np.stack(states).astype(np.float32)
    action_array = np.stack(actions).astype(np.float32)
    metadata = pd.DataFrame(metadata_rows)
    if len(metadata) != extracted or state_array.shape != (extracted, 26):
        raise RuntimeError(
            f"Metadata mismatch: rows={len(metadata)}, states={state_array.shape}, extracted={extracted}"
        )
    if action_array.shape != (extracted, 19):
        raise RuntimeError(f"Unexpected action shape: {action_array.shape}")
    if metadata[["episode_index", "frame_index"]].duplicated().any():
        raise RuntimeError("Duplicate holdout episode/frame samples")
    if not metadata["progress"].between(0.0, 1.0).all():
        raise RuntimeError("Holdout progress is outside [0, 1]")

    metadata.to_csv(working_dir / "metadata.csv", index=False)
    np.savez_compressed(
        working_dir / "robot_metadata.npz",
        state=state_array,
        action=action_array,
    )
    manifest = {
        "schema_version": HOLDOUT_CACHE_SCHEMA,
        "created_at_unix": time.time(),
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": VL_CACHE_FEATURE_TAP,
        "processor_mode": VL_CACHE_PROCESSOR_MODE,
        "dataset_dir": str(dataset_dir),
        "dataset_metadata_sha256": dataset_metadata_sha256,
        "training_dataset_dir": str(training_dataset_dir),
        "training_dataset_metadata_sha256": training_dataset_metadata_sha256,
        "raw_episode_id_overlap": overlap,
        "holdout_raw_episode_ids": len(holdout_ids),
        "training_raw_episode_ids": len(training_ids),
        "num_episodes": int(metadata["episode_index"].nunique()),
        "num_samples": extracted,
        "frame_stride": args.frame_stride,
        "prefix_shape": [192, 2048],
        "cache_dtype": args.cache_dtype,
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 192,
        "transform": "eval",
        "checkpoint_fingerprint": model_fingerprint,
        "checkpoint_files": model_file_hashes,
        "base_model_path": str(base_model_path),
        "processor_path": str(processor_path),
        "vlm_model_path": str(Path(args.vlm_model_path).expanduser().resolve()),
        "modality_config_path": str(Path(args.modality_config_path).expanduser().resolve()),
        "video_modality_keys": image_keys,
        "video_backend": args.video_backend,
        "instruction": args.instruction,
        "seed": args.seed,
        "shards": shards,
        "metadata_csv_sha256": file_sha256(working_dir / "metadata.csv"),
        "robot_metadata_npz_sha256": file_sha256(working_dir / "robot_metadata.npz"),
        "elapsed_seconds": time.time() - started,
    }
    manifest["fingerprint"] = canonical_json_sha256(manifest)
    _write_json(working_dir / "manifest.json", manifest)
    os.replace(working_dir, output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
