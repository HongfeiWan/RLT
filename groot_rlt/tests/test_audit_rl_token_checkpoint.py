from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groot_rlt.integration.artifact_lineage import (
    canonical_json_sha256,
    checkpoint_fingerprint,
    file_sha256,
)
from groot_rlt.representation.audit_rl_token_checkpoint import _tensor_tree_summary, audit


def _make_fake_groot_checkpoint(path: Path) -> tuple[str, dict[str, str]]:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "processor_config.json").write_text("{}", encoding="utf-8")
    (path / "statistics.json").write_text("{}", encoding="utf-8")
    (path / "embodiment_id.json").write_text("{}", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"backbone.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    return checkpoint_fingerprint(path)


def test_audit_accepts_bound_strict_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint-400000"
    model_fingerprint, model_hashes = _make_fake_groot_checkpoint(model_path)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shard_path = cache_dir / "shard_000000.pt"
    torch.save(
        {
            "packed": torch.ones(2, 3, 4, dtype=torch.bfloat16),
            "packed_mask": torch.ones(2, 3, dtype=torch.bool),
            "packed_image_mask": torch.ones(2, 3, dtype=torch.bool),
            "token_counts": [3, 3],
            "selected_counts": [3, 3],
        },
        shard_path,
    )
    cache_manifest = {
        "schema_version": 2,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": "raw_backbone_pre_action_head",
        "processor_mode": "eval",
        "checkpoint_fingerprint": model_fingerprint,
        "checkpoint_files": model_hashes,
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 3,
        "cache_dtype": "bfloat16",
        "num_samples": 2,
        "num_valid_tokens": 6,
        "shards": [
            {
                "file": shard_path.name,
                "sha256": file_sha256(shard_path),
                "num_samples": 2,
                "num_valid_tokens": 6,
            }
        ],
    }
    cache_manifest["fingerprint"] = canonical_json_sha256(cache_manifest)
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest_path.write_text(json.dumps(cache_manifest), encoding="utf-8")

    lineage = {
        "cache_fingerprint": cache_manifest["fingerprint"],
        "checkpoint_fingerprint": model_fingerprint,
        "feature_tap": "raw_backbone_pre_action_head",
        "processor_mode": "eval",
    }
    state = {
        "query_token": torch.zeros(1, 4),
        "encoder_memory_pos": torch.zeros(3, 4),
        "decoder_query": torch.zeros(3, 4),
        "decoder_memory_pos": torch.zeros(1, 4),
    }
    checkpoint_path = tmp_path / "010000.pt"
    torch.save(
        {
            "schema_version": 2,
            "architecture": "openpi_rlt_strict_cross_attention_v1",
            "step": 10_000,
            "last_loss": 0.5,
            "ema_decay": 0.99,
            "autoencoder": state,
            "autoencoder_raw": {name: value.clone() for name, value in state.items()},
            "optimizer": {"state": {0: {"exp_avg": torch.zeros(4)}}},
            "autoencoder_config": {
                "input_dim": 4,
                "model_dim": 4,
                "rl_token_dim": 4,
                "max_vl_tokens": 3,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "use_prefix_mask_token": False,
                "use_decoder_cross_attention": True,
            },
            "args": {
                "token_scope": "image",
                "token_sampling": "uniform",
                "max_vl_tokens": 3,
                "cache_dtype": "bfloat16",
                "decoder_cross_attention": True,
                "decoder_prefix_corruption": False,
                "autoencoder_bf16": False,
                "representation_lineage": lineage,
            },
        },
        checkpoint_path,
    )

    result = audit(
        argparse.Namespace(
            checkpoint=str(checkpoint_path),
            cache_manifest=str(cache_manifest_path),
            model_path=str(model_path),
            expected_step=10_000,
            expected_token_scope="image",
            expected_token_sampling="uniform",
            expected_max_vl_tokens=3,
            expected_model_dim=4,
            expected_cache_dtype="bfloat16",
            verify_optimizer_finite=True,
            verify_cache_sha256=True,
        )
    )

    assert result["verdict"] == "pass"
    assert result["checkpoint"]["ema"]["all_finite"]
    assert result["checkpoint"]["optimizer"]["all_finite"]
    assert result["prefix_cache"]["all_shard_sha256_verified"]


def test_audit_rejects_nonfinite_weights(tmp_path: Path) -> None:
    value = {"weight": torch.tensor([float("nan")])}

    try:
        _tensor_tree_summary(value, label="test")
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("Audit accepted a non-finite model tensor")
