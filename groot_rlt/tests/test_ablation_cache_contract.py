from __future__ import annotations

import json
from pathlib import Path

from groot_rlt.integration.artifact_lineage import canonical_json_sha256
from groot_rlt.representation.evaluate_vl_embedding_autoencoder_ablation import (
    validate_evaluation_cache_contract,
)


def test_ablation_cache_is_bound_to_checkpoint_lineage(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 2,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": "raw_backbone_pre_action_head",
        "processor_mode": "eval",
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 192,
        "cache_dtype": "bfloat16",
        "episode_sampling_rate": 1.0,
        "dataset_sampling_seed": 42,
        "checkpoint_fingerprint": "sha256:model",
    }
    manifest["fingerprint"] = canonical_json_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint_args = {
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 192,
        "cache_dtype": "bfloat16",
        "episode_sampling_rate": 1.0,
        "seed": 42,
        "representation_lineage": {
            "cache_fingerprint": manifest["fingerprint"],
            "checkpoint_fingerprint": "sha256:model",
            "feature_tap": "raw_backbone_pre_action_head",
            "processor_mode": "eval",
        },
    }

    actual = validate_evaluation_cache_contract(tmp_path, checkpoint_args)
    assert actual["fingerprint"] == manifest["fingerprint"]

    checkpoint_args["representation_lineage"]["cache_fingerprint"] = "sha256:wrong"
    try:
        validate_evaluation_cache_contract(tmp_path, checkpoint_args)
    except ValueError as exc:
        assert "lineage mismatch" in str(exc)
    else:
        raise AssertionError("Ablation accepted a cache outside checkpoint lineage")
