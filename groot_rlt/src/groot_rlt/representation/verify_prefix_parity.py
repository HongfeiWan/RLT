#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Verify cache-time and serving-time GR00T prefix extraction parity."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from groot_rlt.groot_repo import ensure_groot_repo

REPO_ROOT = ensure_groot_repo()

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.types import MessageType, VLAStepData  # noqa: E402

from groot_rlt.integration.checkpoint_policy_utils import (  # noqa: E402
    load_checkpoint_modality_config,
    resolve_processor_path,
)
from groot_rlt.integration.lerobot_policy_helpers import (  # noqa: E402
    _build_observation,
    _extract_groups,
    _feature_dim,
    _get_instruction,
    _load_episode_parquet,
    _read_json,
    _to_matrix,
)
from groot_rlt.representation.precompute_rl_tokens_and_vla_actions import (  # noqa: E402
    PrecomputeCheckpointRokaePolicy,
    concat_batched_action_dict,
)
from groot_rlt.representation.train_vl_embedding_autoencoder import (  # noqa: E402
    autocast_context,
    build_backbone,
    build_dataset_and_processor,
    move_backbone_inputs,
    pack_vl_tokens,
)
from groot_rlt.representation.visualize_rl_token_umap import (  # noqa: E402
    load_rl_token_encoder,
)


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-repo-path", type=str, default=str(REPO_ROOT))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--vlm-model-path", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--modality-config-path", type=str, required=True)
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    parser.add_argument("--token-scope", choices=("all", "image", "non_image"), default="image")
    parser.add_argument("--token-sampling", choices=("head", "tail", "uniform"), default="uniform")
    parser.add_argument("--max-vl-tokens", type=int, default=192)
    parser.add_argument(
        "--cache-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--rl-token-checkpoint", type=str, default=None)
    parser.add_argument("--denoise-steps", type=int, default=32)
    parser.add_argument("--verify-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def _training_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_path=args.model_path,
        processor_path=args.processor_path,
        vlm_model_path=args.vlm_model_path,
        trust_remote_code=True,
        local_files_only=True,
        load_bf16=True,
        use_flash_attention=None,
        modality_config_path=args.modality_config_path,
        embodiment_tag=args.embodiment_tag,
        dataset_dir=args.dataset_dir,
        video_backend=args.video_backend,
        episode_sampling_rate=1.0,
        seed=args.seed,
        instruction=args.instruction,
    )


_VL_BACKBONE_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
)


def _tensor_snapshot(
    values: Any,
    keys: tuple[str, ...] = _VL_BACKBONE_INPUT_KEYS,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in dict(values).items()
        if key in keys and isinstance(value, torch.Tensor)
    }


def _extract_training_path(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    train_args = _training_args(args)
    backbone, model_cfg, loading_kwargs = build_backbone(train_args, device)
    dataset, processor = build_dataset_and_processor(train_args, model_cfg, loading_kwargs)
    processor.eval()

    episode = dataset.loader[int(args.episode_index)]
    row = episode.iloc[int(args.frame_index)]
    image_keys = dataset.video_modality.modality_keys
    language_key = dataset.language_modality.modality_keys[0]
    language = args.instruction or str(row[f"language.{language_key}"])
    if processor.formalize_language:
        language = re.sub(r"[^\w\s]", "", language.lower())
    feature = processor._get_vlm_inputs(
        image_keys=image_keys,
        images={key: [row[f"video.{key}"]] for key in image_keys},
        masks=None,
        image_transform=processor.eval_image_transform,
        language=language,
    )
    batch = processor.collator([feature])
    dtype = next(backbone.parameters()).dtype
    backbone_inputs = move_backbone_inputs(batch["inputs"], device, dtype)
    with torch.inference_mode(), autocast_context(device, dtype == torch.bfloat16):
        output = backbone(backbone_inputs)
        packed, packed_mask, packed_image_mask, counts, selected = pack_vl_tokens(
            output,
            token_scope=args.token_scope,
            max_tokens=args.max_vl_tokens,
            token_sampling=args.token_sampling,
        )
    tensors = {
        "backbone_features": output["backbone_features"].detach().cpu(),
        "backbone_attention_mask": output["backbone_attention_mask"].detach().cpu(),
        "image_mask": output["image_mask"].detach().cpu(),
        "packed": packed.detach().cpu(),
        "packed_mask": packed_mask.detach().cpu(),
        "packed_image_mask": packed_image_mask.detach().cpu(),
    }
    metadata = {
        "language": language,
        "token_counts": counts,
        "selected_counts": selected,
    }
    inputs = _tensor_snapshot(backbone_inputs)
    del output, backbone, processor, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return inputs, tensors, metadata


def _serving_observation(
    args: argparse.Namespace,
    modality_config: dict[str, Any],
) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    modality_meta = _read_json(dataset_dir / "meta" / "modality.json")
    frame_table = _load_episode_parquet(dataset_dir, int(args.episode_index))
    states = _to_matrix(
        frame_table["observation.state"],
        expected_dim=_feature_dim(dataset_dir, "observation.state"),
        name="observation.state",
    )
    states_by_key = _extract_groups(
        states,
        modality_meta,
        "state",
        list(modality_config["state"].modality_keys),
    )
    instruction = _get_instruction(dataset_dir, frame_table, override=args.instruction)
    return _build_observation(
        dataset_dir,
        episode_index=int(args.episode_index),
        step=int(args.frame_index),
        states_by_key=states_by_key,
        modality_config=modality_config,
        modality_meta=modality_meta,
        instruction=instruction,
        video_backend=args.video_backend,
    )


def _extract_serving_path(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
    PrecomputeCheckpointRokaePolicy,
    dict[str, Any],
]:
    model_path = Path(args.model_path).expanduser().resolve()
    processor_path = (
        Path(args.processor_path).expanduser().resolve()
        if args.processor_path is not None
        else resolve_processor_path(model_path)
    )
    embodiment = EmbodimentTag.resolve(args.embodiment_tag)
    modality_config = load_checkpoint_modality_config(model_path, embodiment)
    policy = PrecomputeCheckpointRokaePolicy(
        model_path=model_path,
        processor_path=processor_path,
        device=str(device),
        strict=True,
        vlm_model_path=Path(args.vlm_model_path).expanduser().resolve(),
        embodiment_tag=embodiment,
    )
    observation = _serving_observation(args, modality_config)
    item = policy._unbatch_observation(observation)[0]
    step = VLAStepData(
        images=item["video"],
        states=item["state"],
        actions={},
        text=item["language"][policy.language_key][0],
        embodiment=policy.embodiment_tag,
    )
    processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    collated = policy.collate_fn([processed])
    collated = policy._rec_to_dtype(collated, dtype=torch.bfloat16)
    with torch.inference_mode():
        backbone_inputs, _ = policy.model.prepare_input(**collated)
        output = policy.model.backbone(backbone_inputs)
        packed, packed_mask, packed_image_mask, counts, selected = pack_vl_tokens(
            output,
            token_scope=args.token_scope,
            max_tokens=args.max_vl_tokens,
            token_sampling=args.token_sampling,
        )
    tensors = {
        "backbone_features": output["backbone_features"].detach().cpu(),
        "backbone_attention_mask": output["backbone_attention_mask"].detach().cpu(),
        "image_mask": output["image_mask"].detach().cpu(),
        "packed": packed.detach().cpu(),
        "packed_mask": packed_mask.detach().cpu(),
        "packed_image_mask": packed_image_mask.detach().cpu(),
    }
    metadata = {
        "token_counts": counts,
        "selected_counts": selected,
    }
    return _tensor_snapshot(backbone_inputs), tensors, metadata, policy, observation


def _compare_tensors(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise AssertionError(
            f"{label} keys differ: training={sorted(left)} serving={sorted(right)}"
        )
    results: dict[str, Any] = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(
                f"{label}.{key} contract differs: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
            )
        equal = torch.equal(a, b)
        max_abs = 0.0
        if torch.is_floating_point(a):
            max_abs = float((a.float() - b.float()).abs().max().item())
        results[key] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "exact": bool(equal),
            "max_abs": max_abs,
        }
        if not equal:
            raise AssertionError(f"{label}.{key} is not bitwise identical; max_abs={max_abs}")
    return results


def main() -> None:
    args = make_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_inputs, train_prefix, train_meta = _extract_training_path(args, device)
    serving_inputs, serving_prefix, serving_meta, policy, observation = _extract_serving_path(
        args, device
    )
    input_results = _compare_tensors(train_inputs, serving_inputs, label="backbone_inputs")
    prefix_results = _compare_tensors(train_prefix, serving_prefix, label="prefix")
    if train_meta["selected_counts"] != serving_meta["selected_counts"]:
        raise AssertionError(
            "Selected token counts differ: "
            f"training={train_meta['selected_counts']} serving={serving_meta['selected_counts']}"
        )
    if serving_prefix["packed"].shape != (1, args.max_vl_tokens, 2048):
        raise AssertionError(
            f"Unexpected packed prefix shape: {tuple(serving_prefix['packed'].shape)}"
        )

    storage_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.cache_dtype]
    cache_row_path = output_dir / "golden_cache_row.pt"
    torch.save(
        {
            "packed": train_prefix["packed"].to(dtype=storage_dtype),
            "packed_mask": train_prefix["packed_mask"],
            "packed_image_mask": train_prefix["packed_image_mask"],
        },
        cache_row_path,
    )
    cache_row = torch.load(cache_row_path, map_location="cpu", weights_only=True)
    serving_cache_view = {
        key: serving_prefix[key]
        for key in ("packed", "packed_mask", "packed_image_mask")
    }
    cache_roundtrip_results = _compare_tensors(
        cache_row,
        serving_cache_view,
        label="cache_roundtrip",
    )

    summary: dict[str, Any] = {
        "verdict": "pass",
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "processor_path": str(
            Path(args.processor_path).expanduser().resolve()
            if args.processor_path
            else resolve_processor_path(Path(args.model_path))
        ),
        "vlm_model_path": str(Path(args.vlm_model_path).expanduser().resolve()),
        "episode_index": args.episode_index,
        "frame_index": args.frame_index,
        "token_scope": args.token_scope,
        "token_sampling": args.token_sampling,
        "max_vl_tokens": args.max_vl_tokens,
        "cache_dtype": args.cache_dtype,
        "input_parity": input_results,
        "prefix_parity": prefix_results,
        "cache_roundtrip_parity": cache_roundtrip_results,
        "training_metadata": train_meta,
        "serving_metadata": serving_meta,
    }

    if args.rl_token_checkpoint is not None:
        encoder, checkpoint_args, step, _ = load_rl_token_encoder(
            Path(args.rl_token_checkpoint).expanduser().resolve(), device
        )
        with torch.inference_mode():
            z_training = encoder.encode_rl_token(
                train_prefix["packed"].to(device=device, dtype=torch.float32),
                train_prefix["packed_mask"].to(device=device),
            ).cpu()
            z_serving = encoder.encode_rl_token(
                serving_prefix["packed"].to(device=device, dtype=torch.float32),
                serving_prefix["packed_mask"].to(device=device),
            ).cpu()
            z_repeat = encoder.encode_rl_token(
                serving_prefix["packed"].to(device=device, dtype=torch.float32),
                serving_prefix["packed_mask"].to(device=device),
            ).cpu()
        if not torch.equal(z_training, z_serving) or not torch.equal(z_serving, z_repeat):
            raise AssertionError(
                "RL-token encoder output is not deterministic and bitwise identical"
            )
        if not torch.isfinite(z_serving).all():
            raise AssertionError("RL-token encoder produced non-finite values")
        summary["rl_token"] = {
            "checkpoint": str(Path(args.rl_token_checkpoint).expanduser().resolve()),
            "checkpoint_step": step,
            "shape": list(z_serving.shape),
            "norm": float(torch.linalg.vector_norm(z_serving.float()).item()),
            "exact": True,
            "training_lineage": checkpoint_args.get("representation_lineage"),
        }

    if args.verify_reference:
        policy.model.action_head.num_inference_timesteps = int(args.denoise_steps)
        torch.manual_seed(args.seed)
        action, _ = policy.get_action(observation)
        action_keys = list(policy.modality_configs["action"].modality_keys)
        reference = concat_batched_action_dict(action, action_keys)[0]
        if reference.shape != (32, 26):
            raise AssertionError(f"Expected decoded reference [32,26], got {reference.shape}")
        if not np.isfinite(reference).all():
            raise AssertionError("Decoded VLA reference contains non-finite values")
        summary["reference"] = {
            "denoise_steps": int(args.denoise_steps),
            "full_shape": list(reference.shape),
            "rlt_source_shape": list(reference[:10].shape),
            "rlt_projected_shape": list(reference[:10, :19].shape),
        }
    else:
        reference = None

    torch.save(
        {
            "backbone_inputs": serving_inputs,
            "prefix": serving_prefix,
            "rl_token": None if "rl_token" not in summary else z_serving,
            "reference": reference,
        },
        output_dir / "golden.pt",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
