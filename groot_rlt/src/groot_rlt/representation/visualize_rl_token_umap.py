#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Extract RL tokens and visualize them with UMAP.

Two input modes are supported:

1. ``--source dataset`` runs the frozen GR00T/Cosmos VLM on LeRobot frames,
   feeds the packed hidden tokens to the trained RLTokenEncoder, and writes
   RL tokens plus metadata such as episode, frame, task, and optional labels.
2. ``--source cache`` reuses precomputed VL embedding shards from
   ``train_vl_embedding_autoencoder.py``. This is much faster and useful for
   smoke tests. When possible, the script reconstructs the cache sampling order
   from the LeRobot metadata so it can still plot episode/frame/progress labels.

Outputs include:

- ``rl_tokens.npy``: [N, D] RL tokens.
- ``embedding_2d.npy``: [N, 2] reducer output.
- ``metadata.csv``: one row per token.
- ``summary.json``: extraction and reducer details.
- ``rl_token_<reducer>_by_<label>.png``: scatter plots for requested labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from groot_rlt.groot_repo import ensure_groot_repo
from groot_rlt.paths import VL_EMBEDDING_CACHE_DIR

REPO_ROOT = ensure_groot_repo()

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

from groot_rlt.integration.defaults import (  # noqa: E402
    L10_BASE_MODEL_PATH,
    L10_MODALITY_CONFIG_PATH,
    L10_PREPARED_DATASET_DIR,
    L10_VLM_MODEL_PATH,
)
from groot_rlt.representation.train_vl_embedding_autoencoder import (  # noqa: E402
    VLTokenAutoencoder,
    VLTokenAutoencoderConfig,
    autocast_context,
    build_backbone,
    build_dataset_and_processor,
    load_autoencoder_state_dict,
    move_backbone_inputs,
    pack_vl_tokens,
    resolve_path,
    validate_strict_checkpoint_payload,
)

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "vl_embedding_autoencoder_pi_cached"
DEFAULT_CACHE_DIR = VL_EMBEDDING_CACHE_DIR
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "rl_token_umap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groot-repo-path",
        type=str,
        default=str(REPO_ROOT),
        help="Isaac-GR00T checkout used for models, examples, data, and default paths.",
    )
    parser.add_argument("--source", choices=("dataset", "cache"), default="dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="RLTokenEncoder/autoencoder checkpoint. Defaults to latest numeric .pt in --checkpoint-dir.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--embedding-cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--dataset-dir", type=str, default=str(L10_PREPARED_DATASET_DIR))
    parser.add_argument("--embodiment-tag", type=str, default=EmbodimentTag.NEW_EMBODIMENT.value)
    parser.add_argument("--modality-config-path", type=str, default=str(L10_MODALITY_CONFIG_PATH))
    parser.add_argument("--base-model-path", type=str, default=str(L10_BASE_MODEL_PATH))
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--vlm-model-path", type=str, default=str(L10_VLM_MODEL_PATH))
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    parser.add_argument("--episode-indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-sampling-rate",
        type=float,
        default=1.0,
        help=(
            "Fraction of each episode exposed to the dataset loader. Dataset-source "
            "visualization defaults to every frame before --frame-stride is applied."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument(
        "--keyframe-column",
        type=str,
        default=None,
        help="Optional dataframe column used to select key timesteps.",
    )
    parser.add_argument(
        "--keyframe-values",
        type=str,
        nargs="*",
        default=None,
        help="Optional allowed values for --keyframe-column. If omitted, truthy values are kept.",
    )

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--cache-progress-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --source cache, reconstruct episode/frame/progress labels from the "
            "dataset metadata and cache sampling parameters."
        ),
    )
    parser.add_argument(
        "--cache-seed",
        type=int,
        default=None,
        help="Seed used when precomputing the cache. Defaults to checkpoint args seed.",
    )
    parser.add_argument(
        "--cache-episode-sampling-rate",
        type=float,
        default=None,
        help=(
            "Episode sampling rate used when precomputing the cache. Defaults to checkpoint "
            "args episode_sampling_rate."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--autoencoder-bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--use-flash-attention", action=argparse.BooleanOptionalAction, default=None
    )

    parser.add_argument("--token-scope", choices=("all", "image", "non_image"), default=None)
    parser.add_argument(
        "--token-sampling",
        choices=("head", "tail", "uniform", "random"),
        default=None,
    )
    parser.add_argument("--max-vl-tokens", type=int, default=None)

    parser.add_argument("--reducer", choices=("umap", "pca"), default="umap")
    parser.add_argument("--allow-pca-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--umap-n-neighbors", type=int, default=10)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument(
        "--color-by",
        type=str,
        default=(
            "progress_percent,progress_bin,episode_index,task,success,phase,"
            "intervention,validity,reward,shard_index"
        ),
        help="Comma-separated metadata fields to plot.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def checkpoint_step_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"(?:checkpoint_step_)?(\d+)\.pt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for path in checkpoint_dir.glob("*.pt"):
        step = checkpoint_step_from_path(path)
        if step is None:
            continue
        candidates.append((step, path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError(f"No numeric checkpoint .pt files found under: {checkpoint_dir}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_rl_token_encoder(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VLTokenAutoencoder, dict[str, Any], int, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_strict_checkpoint_payload(ckpt)
    ckpt_args = dict(ckpt.get("args", {}))
    config = VLTokenAutoencoderConfig(**ckpt["autoencoder_config"])
    model = VLTokenAutoencoder(config)
    load_autoencoder_state_dict(model, ckpt["autoencoder"])
    step = int(ckpt.get("step", checkpoint_step_from_path(checkpoint_path) or -1))
    last_loss = float(ckpt.get("last_loss", math.nan))
    del ckpt
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, ckpt_args, step, last_loss


def normalize_optional_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (np.generic,)):
        value = value.item()
    return str(value)


def first_existing(row: Any, names: list[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def progress_bin(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def row_metadata(
    row: Any,
    *,
    loader_episode_index: int,
    step_index: int,
    task_text: str,
    episode_length: int,
) -> dict[str, Any]:
    frame_index = int(row["frame_index"]) if "frame_index" in row else int(step_index)
    progress = frame_index / max(episode_length - 1, 1)
    reward = first_existing(row, ["next.reward", "reward"])
    phase = first_existing(
        row,
        [
            "phase",
            "stage",
            "subtask",
            "sub_task",
            "annotation.phase",
            "annotation.stage",
            "annotation.human.phase",
            "annotation.human.stage",
        ],
    )
    intervention = first_existing(
        row,
        [
            "intervention",
            "human_intervention",
            "human_takeover",
            "takeover",
            "teleop",
            "annotation.human.intervention",
            "annotation.human.takeover",
        ],
    )
    success = first_existing(
        row,
        [
            "success",
            "succeeded",
            "episode_success",
            "next.success",
            "annotation.success",
            "annotation.human.success",
        ],
    )
    validity = first_existing(row, ["annotation.human.validity", "validity"])
    return {
        "source": "dataset",
        "loader_episode_index": int(loader_episode_index),
        "episode_index": int(row["episode_index"])
        if "episode_index" in row
        else loader_episode_index,
        "frame_index": frame_index,
        "timestep": int(step_index),
        "timestamp": float(row["timestamp"]) if "timestamp" in row else float("nan"),
        "progress": float(progress),
        "progress_percent": float(progress * 100.0),
        "progress_bin": progress_bin(progress),
        "task_index": normalize_optional_label(row["task_index"] if "task_index" in row else None),
        "task": task_text,
        "success": normalize_optional_label(success),
        "phase": normalize_optional_label(phase) or progress_bin(progress),
        "intervention": normalize_optional_label(intervention),
        "validity": normalize_optional_label(validity),
        "reward": normalize_optional_label(reward),
    }


def select_steps(df: Any, args: argparse.Namespace) -> list[int]:
    if args.keyframe_column is not None:
        if args.keyframe_column not in df.columns:
            raise KeyError(
                f"--keyframe-column {args.keyframe_column!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        values = df[args.keyframe_column]
        if args.keyframe_values:
            allowed = {str(v) for v in args.keyframe_values}
            indices = [int(i) for i, value in enumerate(values) if str(value) in allowed]
        else:
            indices = [int(i) for i, value in enumerate(values) if bool(value)]
    else:
        stride = max(1, int(args.frame_stride))
        indices = list(range(0, len(df), stride))
    if args.max_frames_per_episode is not None:
        indices = indices[: max(0, int(args.max_frames_per_episode))]
    return indices


def append_encoded_batch(
    *,
    autoencoder: VLTokenAutoencoder,
    packed: torch.Tensor,
    packed_mask: torch.Tensor,
    packed_image_mask: torch.Tensor,
    metadata: list[dict[str, Any]],
    token_counts: list[int],
    selected_counts: list[int],
    output_tokens: list[np.ndarray],
    output_metadata: list[dict[str, Any]],
    device: torch.device,
    autoencoder_bf16: bool,
) -> None:
    packed = packed.to(device=device, dtype=torch.float32, non_blocking=True)
    packed_mask = packed_mask.to(device=device, non_blocking=True).bool()
    packed_image_mask = packed_image_mask.to(device=device, non_blocking=True).bool()
    with torch.no_grad(), autocast_context(device, autoencoder_bf16):
        rl_token = autoencoder.encode_rl_token(packed, packed_mask)
    output_tokens.append(rl_token.detach().float().cpu().numpy())
    image_counts = (packed_mask & packed_image_mask).sum(dim=1).detach().cpu().tolist()
    for idx, meta in enumerate(metadata):
        enriched = dict(meta)
        enriched["token_count"] = int(token_counts[idx])
        enriched["selected_count"] = int(selected_counts[idx])
        enriched["image_token_count"] = int(image_counts[idx])
        output_metadata.append(enriched)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_dataset_dir_for_cache(manifest: dict[str, Any], args: argparse.Namespace) -> Path:
    manifest_dataset = manifest.get("dataset_dir")
    if manifest_dataset:
        return Path(resolve_path(manifest_dataset)).expanduser().resolve()
    return Path(resolve_path(args.dataset_dir)).expanduser().resolve()


def parquet_path_for_episode(
    *,
    dataset_dir: Path,
    data_path_pattern: str,
    chunk_size: int,
    episode_index: int,
) -> Path:
    chunk_idx = episode_index // chunk_size
    return dataset_dir / data_path_pattern.format(
        episode_chunk=chunk_idx,
        episode_index=episode_index,
    )


def row_value(row: Any, key: str) -> Any:
    if key in row:
        return row[key]
    return None


def task_text_from_row(
    row: Any,
    *,
    tasks_map: dict[int, str],
    manifest: dict[str, Any],
) -> str:
    if manifest.get("instruction"):
        return str(manifest["instruction"])
    task_index = row_value(row, "annotation.human.action.task_description")
    if task_index is None:
        task_index = row_value(row, "task_index")
    try:
        return tasks_map.get(int(task_index), str(task_index))
    except (TypeError, ValueError):
        return normalize_optional_label(task_index)


def raw_row_metadata(
    row: Any,
    *,
    loader_episode_index: int,
    episode_index: int,
    step_index: int,
    episode_length: int,
    task_text: str,
) -> dict[str, Any]:
    frame_index = row_value(row, "frame_index")
    frame_index = int(frame_index) if frame_index is not None else int(step_index)
    timestamp = row_value(row, "timestamp")
    progress = frame_index / max(episode_length - 1, 1)
    reward = first_existing(row, ["next.reward", "reward"])
    validity = first_existing(row, ["annotation.human.validity", "validity"])
    return {
        "loader_episode_index": int(loader_episode_index),
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "timestep": int(step_index),
        "timestamp": float(timestamp) if timestamp is not None else float("nan"),
        "progress": float(progress),
        "progress_percent": float(progress * 100.0),
        "progress_bin": progress_bin(progress),
        "task_index": normalize_optional_label(row_value(row, "task_index")),
        "task": task_text,
        "validity": normalize_optional_label(validity),
        "reward": normalize_optional_label(reward),
    }


def reconstruct_cache_metadata(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = resolve_dataset_dir_for_cache(manifest, args)
    meta_dir = dataset_dir / "meta"
    with (meta_dir / "info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    episodes = read_jsonl(meta_dir / "episodes.jsonl")
    tasks = read_jsonl(meta_dir / "tasks.jsonl")
    tasks_map = {int(task["task_index"]): str(task["task"]) for task in tasks}

    seed = int(args.cache_seed if args.cache_seed is not None else args.seed)

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - exercised in local tooling.
        raise RuntimeError("pandas is required to reconstruct cache progress metadata.") from exc

    episode_tables = {}
    total_frames = 0
    for loader_episode_index, episode_meta in enumerate(episodes):
        episode_index = int(episode_meta.get("episode_index", int(loader_episode_index)))
        parquet_path = parquet_path_for_episode(
            dataset_dir=dataset_dir,
            data_path_pattern=info["data_path"],
            chunk_size=int(info["chunks_size"]),
            episode_index=episode_index,
        )
        try:
            episode_df = pd.read_parquet(parquet_path)
        except Exception as exc:  # pragma: no cover - exercised in local tooling.
            raise RuntimeError(f"Failed to read parquet metadata from {parquet_path}") from exc
        episode_length = min(len(episode_df), int(episode_meta.get("length", len(episode_df))))
        episode_tables[int(loader_episode_index)] = (episode_index, episode_df, episode_length)
        total_frames += episode_length

    def build_rows(episode_sampling_rate: float) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        episode_order = np.arange(len(episodes))
        rng.shuffle(episode_order)
        rows: list[dict[str, Any]] = []
        for loader_episode_index in episode_order:
            episode_index, episode_df, episode_length = episode_tables[int(loader_episode_index)]
            step_indices = np.arange(episode_length)
            rng.shuffle(step_indices)
            if episode_sampling_rate < 1.0:
                keep = max(1, int(round(episode_length * episode_sampling_rate)))
                step_indices = step_indices[:keep]
            for step_index in step_indices:
                row = episode_df.iloc[int(step_index)]
                rows.append(
                    raw_row_metadata(
                        row,
                        loader_episode_index=int(loader_episode_index),
                        episode_index=episode_index,
                        step_index=int(step_index),
                        episode_length=episode_length,
                        task_text=task_text_from_row(row, tasks_map=tasks_map, manifest=manifest),
                    )
                )
        return rows

    manifest_samples = int(manifest.get("num_samples", 0))
    candidate_rates: list[float] = []
    if args.cache_episode_sampling_rate is not None:
        candidate_rates.append(float(args.cache_episode_sampling_rate))
    else:
        if manifest.get("episode_sampling_rate") is not None:
            candidate_rates.append(float(manifest["episode_sampling_rate"]))
        candidate_rates.extend([1.0, manifest_samples / max(total_frames, 1), 0.1])

    rows = []
    episode_sampling_rate = candidate_rates[0]
    seen_rates: set[float] = set()
    for candidate_rate in candidate_rates:
        if candidate_rate in seen_rates:
            continue
        seen_rates.add(candidate_rate)
        candidate_rows = build_rows(candidate_rate)
        if not rows:
            rows = candidate_rows
            episode_sampling_rate = candidate_rate
        if len(candidate_rows) == manifest_samples:
            rows = candidate_rows
            episode_sampling_rate = candidate_rate
            break

    info_out = {
        "cache_progress_metadata": True,
        "cache_metadata_dataset_dir": str(dataset_dir),
        "cache_metadata_seed": seed,
        "cache_metadata_episode_sampling_rate": episode_sampling_rate,
        "cache_metadata_rows": len(rows),
        "cache_metadata_candidate_rates": candidate_rates,
        "cache_manifest_samples": manifest_samples,
        "cache_metadata_aligned": len(rows) == manifest_samples,
    }
    return rows, info_out


def extract_from_cache(
    *,
    autoencoder: VLTokenAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
    autoencoder_bf16: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    cache_dir = Path(args.embedding_cache_dir).expanduser().resolve()
    with (cache_dir / "manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    reconstructed_metadata: list[dict[str, Any]] = []
    reconstruction_info: dict[str, Any] = {"cache_progress_metadata": False}
    if args.cache_progress_metadata:
        reconstructed_metadata, reconstruction_info = reconstruct_cache_metadata(manifest, args)
        if len(reconstructed_metadata) != int(
            manifest.get("num_samples", len(reconstructed_metadata))
        ):
            print(
                "Warning: reconstructed cache metadata length does not match manifest; "
                "progress labels may be incomplete.",
                flush=True,
            )

    output_tokens: list[np.ndarray] = []
    output_metadata: list[dict[str, Any]] = []
    global_sample = 0
    max_samples = args.max_samples if args.max_samples is not None else int(manifest["num_samples"])

    for shard_index, shard_info in enumerate(manifest["shards"]):
        if global_sample >= max_samples:
            break
        shard = torch.load(cache_dir / shard_info["file"], map_location="cpu")
        num_samples = int(shard["packed"].shape[0])
        for start in range(0, num_samples, args.batch_size):
            if global_sample >= max_samples:
                break
            end = min(num_samples, start + args.batch_size, start + (max_samples - global_sample))
            packed = shard["packed"][start:end].float()
            packed_mask = shard["packed_mask"][start:end].bool()
            packed_image_mask = shard["packed_image_mask"][start:end].bool()
            token_counts = [int(x) for x in shard["token_counts"][start:end]]
            selected_counts = [int(x) for x in shard["selected_counts"][start:end]]
            batch_metadata = []
            for local_idx in range(start, end):
                absolute_idx = int(global_sample + local_idx - start)
                reconstructed = (
                    dict(reconstructed_metadata[absolute_idx])
                    if absolute_idx < len(reconstructed_metadata)
                    else {}
                )
                reconstructed.setdefault("episode_index", "")
                reconstructed.setdefault("frame_index", "")
                reconstructed.setdefault(
                    "task", normalize_optional_label(manifest.get("instruction"))
                )
                reconstructed.setdefault("progress", "")
                reconstructed.setdefault("progress_percent", "")
                reconstructed.setdefault("progress_bin", "")
                batch_metadata.append(
                    {
                        **reconstructed,
                        "source": "cache",
                        "sample_index": absolute_idx,
                        "shard_index": int(shard_index),
                        "shard_file": shard_info["file"],
                        "cache_local_index": int(local_idx),
                    }
                )
            append_encoded_batch(
                autoencoder=autoencoder,
                packed=packed,
                packed_mask=packed_mask,
                packed_image_mask=packed_image_mask,
                metadata=batch_metadata,
                token_counts=token_counts,
                selected_counts=selected_counts,
                output_tokens=output_tokens,
                output_metadata=output_metadata,
                device=device,
                autoencoder_bf16=autoencoder_bf16,
            )
            global_sample += end - start
        del shard

    if not output_tokens:
        raise RuntimeError("No RL tokens were extracted from cache.")
    return (
        np.concatenate(output_tokens, axis=0),
        output_metadata,
        {"cache_manifest": manifest, **reconstruction_info},
    )


def extract_from_dataset(
    *,
    autoencoder: VLTokenAutoencoder,
    args: argparse.Namespace,
    device: torch.device,
    autoencoder_bf16: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    backbone, model_cfg, transformers_loading_kwargs = build_backbone(args, device)
    dataset, processor = build_dataset_and_processor(args, model_cfg, transformers_loading_kwargs)
    processor.eval()

    loader = dataset.loader
    image_keys = dataset.video_modality.modality_keys
    language_key = dataset.language_modality.modality_keys[0]
    language_column = f"language.{language_key}"

    if args.episode_indices is None:
        episode_indices = list(range(len(loader)))
    else:
        episode_indices = [int(i) for i in args.episode_indices]
    if args.max_episodes is not None:
        episode_indices = episode_indices[: max(0, int(args.max_episodes))]

    output_tokens: list[np.ndarray] = []
    output_metadata: list[dict[str, Any]] = []
    pending_features: list[dict[str, Any]] = []
    pending_metadata: list[dict[str, Any]] = []
    extracted = 0

    def flush() -> None:
        nonlocal pending_features, pending_metadata
        if not pending_features:
            return
        batch = processor.collator(pending_features)
        inputs = batch["inputs"]
        dtype = next(backbone.parameters()).dtype
        backbone_inputs = move_backbone_inputs(inputs, device, dtype)
        with torch.no_grad(), autocast_context(device, dtype == torch.bfloat16):
            backbone_output = backbone(backbone_inputs)
        packed, packed_mask, packed_image_mask, token_counts, selected_counts = pack_vl_tokens(
            backbone_output,
            token_scope=args.token_scope,
            max_tokens=args.max_vl_tokens,
            token_sampling=args.token_sampling,
        )
        append_encoded_batch(
            autoencoder=autoencoder,
            packed=packed.detach().float(),
            packed_mask=packed_mask.detach(),
            packed_image_mask=packed_image_mask.detach(),
            metadata=pending_metadata,
            token_counts=token_counts,
            selected_counts=selected_counts,
            output_tokens=output_tokens,
            output_metadata=output_metadata,
            device=device,
            autoencoder_bf16=autoencoder_bf16,
        )
        pending_features = []
        pending_metadata = []

    for loader_episode_index in episode_indices:
        if args.max_samples is not None and extracted >= args.max_samples:
            break
        episode = loader[int(loader_episode_index)]
        for step_index in select_steps(episode, args):
            if args.max_samples is not None and extracted >= args.max_samples:
                break
            row = episode.iloc[int(step_index)]
            images = {key: [row[f"video.{key}"]] for key in image_keys}
            language = args.instruction or str(row[language_column])
            if processor.formalize_language:
                language = re.sub(r"[^\w\s]", "", language.lower())
            feature = processor._get_vlm_inputs(
                image_keys=image_keys,
                images=images,
                masks=None,
                image_transform=processor.eval_image_transform,
                language=language,
            )
            pending_features.append(feature)
            pending_metadata.append(
                row_metadata(
                    row,
                    loader_episode_index=int(loader_episode_index),
                    step_index=int(step_index),
                    task_text=language,
                    episode_length=len(episode),
                )
            )
            extracted += 1
            if len(pending_features) >= args.batch_size:
                flush()
    flush()

    if not output_tokens:
        raise RuntimeError("No RL tokens were extracted from dataset.")
    return (
        np.concatenate(output_tokens, axis=0),
        output_metadata,
        {
            "backbone_model_name": model_cfg.model_name,
            "dataset_episodes": episode_indices,
        },
    )


def reduce_tokens(tokens: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if tokens.shape[0] < 2:
        raise ValueError("Need at least two RL tokens for 2D visualization.")
    x = tokens.astype(np.float32, copy=False)
    if args.reducer == "umap":
        if importlib.util.find_spec("umap") is not None:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(args.umap_n_neighbors, max(2, x.shape[0] - 1)),
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                random_state=args.seed,
            )
            return reducer.fit_transform(x), "umap"
        if not args.allow_pca_fallback:
            raise ImportError(
                "umap-learn is not installed. Install it with: uv pip install umap-learn"
            )
        print("umap-learn is not installed; falling back to PCA for this run.", flush=True)

    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ vh[:2].T, "pca"


def write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_optional_label(row.get(key)) for key in fieldnames})


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "label"


def values_for_label(rows: list[dict[str, Any]], label: str) -> list[str]:
    return [normalize_optional_label(row.get(label)) for row in rows]


def maybe_numeric(values: list[str]) -> np.ndarray | None:
    converted = []
    for value in values:
        if value == "":
            converted.append(np.nan)
            continue
        try:
            converted.append(float(value))
        except ValueError:
            return None
    return np.asarray(converted, dtype=np.float32)


def plot_label(
    *,
    embedding: np.ndarray,
    rows: list[dict[str, Any]],
    label: str,
    output_dir: Path,
    reducer_name: str,
    dpi: int,
) -> Path | None:
    values = values_for_label(rows, label)
    non_empty = [value for value in values if value != ""]
    if not non_empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    numeric = maybe_numeric(values)
    unique_values = sorted(set(non_empty), key=str)
    title = f"RL token {reducer_name.upper()} by {label}"
    if numeric is not None and len(unique_values) > 20:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=numeric,
            s=14,
            alpha=0.78,
            cmap="viridis",
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label=label)
    else:
        cmap = plt.get_cmap("tab20", max(len(unique_values), 1))
        color_map = {value: cmap(i % 20) for i, value in enumerate(unique_values)}
        for value in unique_values:
            indices = np.asarray([item == value for item in values], dtype=bool)
            ax.scatter(
                embedding[indices, 0],
                embedding[indices, 1],
                s=16,
                alpha=0.78,
                label=value,
                color=color_map[value],
                linewidths=0,
            )
        if len(unique_values) <= 20:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.set_title(title)
    ax.set_xlabel(f"{reducer_name.upper()}-1")
    ax.set_ylabel(f"{reducer_name.upper()}-2")
    ax.grid(True, alpha=0.2)
    output_path = output_dir / f"rl_token_{reducer_name}_by_{safe_name(label)}.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else latest_checkpoint(Path(args.checkpoint_dir).expanduser().resolve())
    )
    autoencoder, ckpt_args, checkpoint_step, checkpoint_loss = load_rl_token_encoder(
        checkpoint,
        device,
    )
    args.token_scope = args.token_scope or ckpt_args.get("token_scope", "all")
    args.token_sampling = args.token_sampling or ckpt_args.get("token_sampling", "uniform")
    args.max_vl_tokens = int(args.max_vl_tokens or ckpt_args.get("max_vl_tokens", 512))
    args.autoencoder_bf16 = (
        bool(ckpt_args.get("autoencoder_bf16", device.type == "cuda"))
        if args.autoencoder_bf16 is None
        else bool(args.autoencoder_bf16)
    )
    args.autoencoder_bf16 = bool(args.autoencoder_bf16 and device.type == "cuda")
    if args.cache_seed is None:
        args.cache_seed = int(ckpt_args.get("seed", args.seed))

    tic = time.time()
    if args.source == "cache":
        tokens, metadata, extra = extract_from_cache(
            autoencoder=autoencoder,
            args=args,
            device=device,
            autoencoder_bf16=args.autoencoder_bf16,
        )
    else:
        tokens, metadata, extra = extract_from_dataset(
            autoencoder=autoencoder,
            args=args,
            device=device,
            autoencoder_bf16=args.autoencoder_bf16,
        )

    embedding, reducer_used = reduce_tokens(tokens, args)
    np.save(output_dir / "rl_tokens.npy", tokens)
    np.save(output_dir / "embedding_2d.npy", embedding.astype(np.float32))
    write_metadata_csv(output_dir / "metadata.csv", metadata)

    requested_labels = [label.strip() for label in args.color_by.split(",") if label.strip()]
    plot_paths = []
    for label in requested_labels:
        path = plot_label(
            embedding=embedding,
            rows=metadata,
            label=label,
            output_dir=output_dir,
            reducer_name=reducer_used,
            dpi=args.dpi,
        )
        if path is not None:
            plot_paths.append(str(path))

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_last_loss": checkpoint_loss,
        "source": args.source,
        "device": str(device),
        "autoencoder_bf16": bool(args.autoencoder_bf16),
        "num_tokens": int(tokens.shape[0]),
        "token_dim": int(tokens.shape[1]),
        "reducer_requested": args.reducer,
        "reducer_used": reducer_used,
        "output_dir": str(output_dir),
        "plot_paths": plot_paths,
        "elapsed_seconds": float(time.time() - tic),
        "args": vars(args),
        **extra,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_step={checkpoint_step} checkpoint_last_loss={checkpoint_loss:.6f}")
    print(f"source={args.source} tokens={tokens.shape[0]} dim={tokens.shape[1]}")
    print(f"reducer={reducer_used} output_dir={output_dir}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
