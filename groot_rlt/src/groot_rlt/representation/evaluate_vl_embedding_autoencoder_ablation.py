#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Ablation gate for VL embedding autoencoder checkpoints.

This script checks whether the compact RL token is merely non-zero, or whether
the decoder actually depends on the sample-specific VLM information encoded in it.

The main pass/fail signal is:

- ``zero`` z_rl should be worse than normal z_rl.
- ``batch_shuffle`` z_rl should be worse than normal z_rl.
- ``image_zero_z`` should be worse than ``normal_z`` when decoding the original
  VL tokens.

If shuffle/image-zero do not hurt reconstruction, the checkpoint may have learned
reconstruction from learned position queries without using the correct sample's z_rl content.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from groot_rlt.groot_repo import ensure_groot_repo
from groot_rlt.paths import VL_EMBEDDING_CACHE_DIR

REPO_ROOT = ensure_groot_repo()

from groot_rlt.representation.train_vl_embedding_autoencoder import (  # noqa: E402
    CachedVLEmbeddingDataset,
    VLTokenAutoencoder,
    VLTokenAutoencoderConfig,
    autocast_context,
    load_autoencoder_state_dict,
    make_zrl_ablation_variants,
    masked_mse_loss,
    reconstruction_metrics,
    validate_strict_checkpoint_payload,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "vl_embedding_autoencoder_pi_cached"
DEFAULT_CACHE_DIR = VL_EMBEDDING_CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groot-repo-path",
        type=str,
        default=str(REPO_ROOT),
        help="Isaac-GR00T checkout used for models, examples, data, and default paths.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt file. Defaults to the latest numeric checkpoint under --checkpoint-dir.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding-cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Defaults to checkpoint args ablation_eval_batches, or 32.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to checkpoint args ablation_batch_size, or 16.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Defaults to checkpoint args ablation_noise_std, or 1.0.",
    )
    parser.add_argument(
        "--autoencoder-bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to checkpoint args autoencoder_bf16, or CUDA-enabled.",
    )
    parser.add_argument("--seed-offset", type=int, default=400_000)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--fail-on-not-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit with status 1 when the ablation gate does not pass.",
    )

    parser.add_argument(
        "--min-zero-ratio",
        type=float,
        default=1.5,
        help="Required zero-z loss ratio vs normal-z.",
    )
    parser.add_argument(
        "--min-shuffle-ratio",
        type=float,
        default=1.05,
        help="Required batch-shuffled z loss ratio vs normal-z.",
    )
    parser.add_argument(
        "--min-image-zero-ratio",
        type=float,
        default=1.05,
        help="Required image-zero encoded z loss ratio vs normal-z.",
    )
    parser.add_argument(
        "--max-image-zero-encoder-cos",
        type=float,
        default=0.5,
        help="Required max cosine(normal z, image-zero z) for encoder image sensitivity.",
    )
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


def new_bucket() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "cosine_sum": 0.0,
        "relative_mse_sum": 0.0,
        "first_mse_sum": 0.0,
        "first_cosine_sum": 0.0,
    }


def update_bucket(
    bucket: dict[str, float],
    *,
    name: str,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    first_mask: torch.Tensor,
    valid_tokens: int,
) -> None:
    loss = masked_mse_loss(reconstruction, target, mask)
    metrics = reconstruction_metrics(reconstruction, target, mask, name)
    bucket["loss_sum"] += float(loss.detach().cpu()) * valid_tokens
    bucket["cosine_sum"] += float(metrics[f"{name}/cosine_similarity"]) * valid_tokens
    bucket["relative_mse_sum"] += float(metrics[f"{name}/relative_mse"]) * valid_tokens

    first_reconstruction = reconstruction[:, 0, :].detach().float()
    first_target = target[:, 0, :].detach().float()
    first_mask_f = first_mask.detach().float()
    first_mse = ((first_reconstruction - first_target).pow(2).mean(dim=-1) * first_mask_f).sum()
    first_cosine = (
        F.cosine_similarity(first_reconstruction, first_target, dim=-1) * first_mask_f
    ).sum()
    bucket["first_mse_sum"] += float(first_mse.cpu())
    bucket["first_cosine_sum"] += float(first_cosine.cpu())


def finalize_buckets(
    buckets: dict[str, dict[str, float]],
    *,
    token_denom: int,
    first_token_denom: int,
    normal_key: str,
    ratio_name: str,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    token_denom = max(token_denom, 1)
    first_token_denom = max(first_token_denom, 1)

    normal_mse = buckets[normal_key]["loss_sum"] / token_denom
    normal_first_mse = buckets[normal_key]["first_mse_sum"] / first_token_denom
    for name, bucket in buckets.items():
        mse = bucket["loss_sum"] / token_denom
        first_mse = bucket["first_mse_sum"] / first_token_denom
        results[name] = {
            "mse": float(mse),
            ratio_name: float(mse / normal_mse if normal_mse > 0 else math.nan),
            "cosine_similarity": float(bucket["cosine_sum"] / token_denom),
            "relative_mse": float(bucket["relative_mse_sum"] / token_denom),
            "first_token_mse": float(first_mse),
            f"first_{ratio_name}": float(
                first_mse / normal_first_mse if normal_first_mse > 0 else math.nan
            ),
            "first_token_cosine": float(bucket["first_cosine_sum"] / first_token_denom),
        }
    return results


def print_table(title: str, rows: dict[str, dict[str, float]], ratio_name: str) -> None:
    print(f"--- {title} ---")
    print(
        f"{'variant':<18} {'mse':>12} {'ratio':>8} {'cosine':>10} "
        f"{'rel_mse':>10} {'first_mse':>14} {'first_ratio':>12} {'first_cos':>10}"
    )
    for name, metrics in rows.items():
        print(
            f"{name:<18} "
            f"{metrics['mse']:12.6f} "
            f"{metrics[ratio_name]:8.3f} "
            f"{metrics['cosine_similarity']:10.6f} "
            f"{metrics['relative_mse']:10.6f} "
            f"{metrics['first_token_mse']:14.6f} "
            f"{metrics[f'first_{ratio_name}']:12.3f} "
            f"{metrics['first_token_cosine']:10.6f}"
        )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else latest_checkpoint(Path(args.checkpoint_dir).expanduser().resolve())
    )
    cache_dir = Path(args.embedding_cache_dir).expanduser().resolve()

    tic = time.time()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_strict_checkpoint_payload(ckpt)
    ckpt_args = dict(ckpt.get("args", {}))
    autoencoder_config = VLTokenAutoencoderConfig(**ckpt["autoencoder_config"])
    autoencoder = VLTokenAutoencoder(autoencoder_config)
    load_autoencoder_state_dict(autoencoder, ckpt["autoencoder"])
    ckpt_step = int(ckpt.get("step", checkpoint_step_from_path(checkpoint) or -1))
    ckpt_loss = float(ckpt.get("last_loss", math.nan))
    del ckpt

    autoencoder.to(device=device, dtype=torch.float32)
    autoencoder.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    eval_batches = int(args.eval_batches or ckpt_args.get("ablation_eval_batches") or 32)
    batch_size = int(args.batch_size or ckpt_args.get("ablation_batch_size") or 16)
    noise_std = float(args.noise_std or ckpt_args.get("ablation_noise_std") or 1.0)
    autoencoder_bf16 = (
        bool(ckpt_args.get("autoencoder_bf16", device.type == "cuda"))
        if args.autoencoder_bf16 is None
        else bool(args.autoencoder_bf16)
    )
    autoencoder_bf16 = autoencoder_bf16 and device.type == "cuda"
    seed = int(ckpt_args.get("seed") or 42)

    data_loader = DataLoader(
        CachedVLEmbeddingDataset(cache_dir, seed=seed + args.seed_offset),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 12_345)

    ablation_names = ("normal", "zero", "batch_shuffle", "noise")
    damaged_names = (
        "normal_z",
        "zero_vector_z",
        "zero_input_z",
        "image_zero_z",
        "non_image_zero_z",
        "image_only_z",
        "non_image_only_z",
    )
    ablation_buckets = {name: new_bucket() for name in ablation_names}
    damaged_buckets = {name: new_bucket() for name in damaged_names}
    encoder_sensitivity = {
        "zero_input": {"cosine_sum": 0.0, "l2_sum": 0.0},
        "image_zero": {"cosine_sum": 0.0, "l2_sum": 0.0},
        "non_image_zero": {"cosine_sum": 0.0, "l2_sum": 0.0},
        "image_only": {"cosine_sum": 0.0, "l2_sum": 0.0},
        "non_image_only": {"cosine_sum": 0.0, "l2_sum": 0.0},
    }

    batches = 0
    samples = 0
    valid_tokens = 0
    first_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            if batches >= eval_batches:
                break
            batches += 1

            packed = batch["packed"].to(device=device, dtype=torch.float32, non_blocking=True)
            packed_mask = batch["packed_mask"].to(device=device, non_blocking=True).bool()
            packed_image_mask = (
                batch["packed_image_mask"].to(device=device, non_blocking=True).bool() & packed_mask
            )
            non_image_mask = packed_mask & ~packed_image_mask
            first_mask = packed_mask[:, 0]

            batch_valid_tokens = int(packed_mask.sum().detach().cpu())
            batch_samples = int(packed.shape[0])
            valid_tokens += batch_valid_tokens
            first_tokens += int(first_mask.sum().detach().cpu())
            samples += batch_samples

            with autocast_context(device, autoencoder_bf16):
                z_rl = autoencoder.encode_rl_token(packed, packed_mask)
                z_ablation_variants = make_zrl_ablation_variants(
                    z_rl,
                    noise_std,
                    generator,
                )
                for name in ablation_names:
                    reconstruction = autoencoder.decode_from_rl_token(
                        z_ablation_variants[name],
                        packed,
                        packed_mask,
                    )
                    update_bucket(
                        ablation_buckets[name],
                        name=name,
                        reconstruction=reconstruction,
                        target=packed,
                        mask=packed_mask,
                        first_mask=first_mask,
                        valid_tokens=batch_valid_tokens,
                    )

                zero_input = torch.zeros_like(packed)
                image_zero = packed.masked_fill(packed_image_mask.unsqueeze(-1), 0.0)
                non_image_zero = packed.masked_fill(non_image_mask.unsqueeze(-1), 0.0)
                image_only = packed.masked_fill(non_image_mask.unsqueeze(-1), 0.0)
                non_image_only = packed.masked_fill(packed_image_mask.unsqueeze(-1), 0.0)
                z_by_input = {
                    "normal_z": z_rl,
                    "zero_vector_z": torch.zeros_like(z_rl),
                    "zero_input_z": autoencoder.encode_rl_token(zero_input, packed_mask),
                    "image_zero_z": autoencoder.encode_rl_token(image_zero, packed_mask),
                    "non_image_zero_z": autoencoder.encode_rl_token(non_image_zero, packed_mask),
                    "image_only_z": autoencoder.encode_rl_token(image_only, packed_mask),
                    "non_image_only_z": autoencoder.encode_rl_token(non_image_only, packed_mask),
                }
                for name in damaged_names:
                    reconstruction = autoencoder.decode_from_rl_token(
                        z_by_input[name],
                        packed,
                        packed_mask,
                    )
                    update_bucket(
                        damaged_buckets[name],
                        name=name,
                        reconstruction=reconstruction,
                        target=packed,
                        mask=packed_mask,
                        first_mask=first_mask,
                        valid_tokens=batch_valid_tokens,
                    )

            z_normal = z_rl.detach().float()
            for key, z_key in (
                ("zero_input", "zero_input_z"),
                ("image_zero", "image_zero_z"),
                ("non_image_zero", "non_image_zero_z"),
                ("image_only", "image_only_z"),
                ("non_image_only", "non_image_only_z"),
            ):
                z_variant = z_by_input[z_key].detach().float()
                encoder_sensitivity[key]["cosine_sum"] += float(
                    F.cosine_similarity(z_normal, z_variant, dim=-1).sum().cpu()
                )
                encoder_sensitivity[key]["l2_sum"] += float(
                    (z_normal - z_variant).norm(dim=-1).sum().cpu()
                )

    ablation = finalize_buckets(
        ablation_buckets,
        token_denom=valid_tokens,
        first_token_denom=first_tokens,
        normal_key="normal",
        ratio_name="ratio_vs_normal",
    )
    damaged_input_z_decode = finalize_buckets(
        damaged_buckets,
        token_denom=valid_tokens,
        first_token_denom=first_tokens,
        normal_key="normal_z",
        ratio_name="ratio_vs_normal_z",
    )
    encoder_input_sensitivity = {
        key: {
            "cos_to_normal": float(value["cosine_sum"] / max(samples, 1)),
            "l2_to_normal": float(value["l2_sum"] / max(samples, 1)),
        }
        for key, value in encoder_sensitivity.items()
    }

    gate_checks = {
        "zero_vector_used": ablation["zero"]["ratio_vs_normal"] >= args.min_zero_ratio,
        "batch_shuffle_hurts": ablation["batch_shuffle"]["ratio_vs_normal"]
        >= args.min_shuffle_ratio,
        "image_zero_z_hurts": damaged_input_z_decode["image_zero_z"]["ratio_vs_normal_z"]
        >= args.min_image_zero_ratio,
        "encoder_image_sensitive": encoder_input_sensitivity["image_zero"]["cos_to_normal"]
        <= args.max_image_zero_encoder_cos,
    }
    ready = all(gate_checks.values())
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt_step,
        "checkpoint_last_loss": ckpt_loss,
        "cache_dir": str(cache_dir),
        "device": str(device),
        "autoencoder_bf16": bool(autoencoder_bf16),
        "eval_batches": int(batches),
        "batch_size": int(batch_size),
        "samples": int(samples),
        "valid_tokens": int(valid_tokens),
        "first_tokens": int(first_tokens),
        "noise_std": float(noise_std),
        "elapsed_seconds": float(time.time() - tic),
        "thresholds": {
            "min_zero_ratio": float(args.min_zero_ratio),
            "min_shuffle_ratio": float(args.min_shuffle_ratio),
            "min_image_zero_ratio": float(args.min_image_zero_ratio),
            "max_image_zero_encoder_cos": float(args.max_image_zero_encoder_cos),
        },
        "gate_checks": gate_checks,
        "ready": bool(ready),
        "ablation": ablation,
        "encoder_input_sensitivity": encoder_input_sensitivity,
        "damaged_input_z_decode": damaged_input_z_decode,
    }

    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_step={ckpt_step} checkpoint_last_loss={ckpt_loss:.6f}")
    print(
        f"device={device} batches={batches} batch_size={batch_size} "
        f"samples={samples} valid_tokens={valid_tokens} autoencoder_bf16={int(autoencoder_bf16)}"
    )
    print_table("z_rl_ablation_query_reconstruction", ablation, "ratio_vs_normal")
    print("--- encoder_input_sensitivity ---")
    for name, metrics in encoder_input_sensitivity.items():
        print(
            f"{name:<18} "
            f"cos_to_normal={metrics['cos_to_normal']:.6f} "
            f"l2_to_normal={metrics['l2_to_normal']:.6f}"
        )
    print_table(
        "damaged_input_z_query_reconstruction",
        damaged_input_z_decode,
        "ratio_vs_normal_z",
    )
    print("--- gate ---")
    for name, passed in gate_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"ready: {'PASS' if ready else 'FAIL'}")

    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote_json={output_json}")

    if args.fail_on_not_ready and not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
