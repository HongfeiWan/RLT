from __future__ import annotations

import copy

import pytest
from rlt_online_rl.inference import _validate_groot_machine_a_metadata
from rlt_online_rl.inference import _validate_machine_a_metadata


def _valid_metadata(*, legacy: bool = False) -> dict:
    source_kind = "legacy_full_checkpoint" if legacy else "encoder_ema_artifact"
    architecture = (
        "openpi_rlt_strict_cross_attention_v1" if legacy else "openpi_rlt_strict_cross_attention_v1.encoder_only"
    )
    return {
        "backend": "groot-n1.7",
        "z_dim": 2048,
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 192,
        "rl_token_contract": {
            "source_kind": source_kind,
            "artifact_kind": ("groot_rlt.legacy_full_training_checkpoint" if legacy else "groot_rlt.encoder_ema"),
            "architecture": architecture,
            "source_architecture": "openpi_rlt_strict_cross_attention_v1",
            "artifact_path": "/artifacts/encoder.pt",
            "artifact_manifest_path": None if legacy else "/artifacts/encoder.pt.manifest.json",
            "artifact_fingerprint": "sha256:" + "1" * 64,
            "artifact_file_sha256": "sha256:" + "2" * 64,
            "encoder_tensor_sha256": "sha256:" + "3" * 64,
            "representation_checkpoint_file_sha256": "sha256:" + "4" * 64,
            "checkpoint_fingerprint": "sha256:" + "5" * 64,
            "cache_fingerprint": "sha256:" + "6" * 64,
            "representation_checkpoint_schema_version": 2,
            "prefix_cache_schema_version": 2,
            "feature_tap": "raw_backbone_pre_action_head",
            "processor_mode": "eval",
            "prefix_cache_manifest_path": "/cache/manifest.json",
            "token_scope": "image",
            "token_sampling": "uniform",
            "max_vl_tokens": 192,
            "input_dim": 2048,
            "z_dim": 2048,
            "video_modality_keys": ["ego_view", "wrist_view"],
            "model_path": "/deployment/checkpoint-400000",
            "processor_path": "/deployment/checkpoint-400000",
            "vlm_model_path": "/deployment/Cosmos-Reason2-2B",
            "vlm_deployment_content_fingerprint": "sha256:" + "7" * 64,
            "vlm_fingerprint_scope": "deployment_only_not_representation_training_lineage",
        },
    }


def test_verified_artifact_and_explicit_legacy_contracts_are_accepted() -> None:
    _validate_groot_machine_a_metadata(_valid_metadata())
    _validate_groot_machine_a_metadata(_valid_metadata(legacy=True))


def test_missing_or_malformed_lineage_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing rl_token_contract"):
        _validate_groot_machine_a_metadata({"backend": "groot-n1.7"})

    bad_sha = _valid_metadata()
    bad_sha["rl_token_contract"]["cache_fingerprint"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_groot_machine_a_metadata(bad_sha)

    random_sampling = _valid_metadata()
    random_sampling["token_sampling"] = "random"
    random_sampling["rl_token_contract"]["token_sampling"] = "random"
    with pytest.raises(ValueError, match="deterministic"):
        _validate_groot_machine_a_metadata(random_sampling)


def test_shape_and_top_level_contract_disagreement_fails_closed() -> None:
    mismatch = _valid_metadata()
    mismatch["z_dim"] = 1024
    with pytest.raises(ValueError, match="conflicts"):
        _validate_groot_machine_a_metadata(mismatch)

    invalid_shape = copy.deepcopy(_valid_metadata())
    invalid_shape["rl_token_contract"]["input_dim"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        _validate_groot_machine_a_metadata(invalid_shape)

    wrong_architecture = _valid_metadata()
    wrong_architecture["rl_token_contract"]["architecture"] = "teacher_forced"
    with pytest.raises(ValueError, match="architecture"):
        _validate_groot_machine_a_metadata(wrong_architecture)


def _pinned_expectation() -> dict:
    return {
        "allow_unpinned": False,
        "expected_backend": "groot-n1.7",
        "expected_checkpoint_fingerprint": "sha256:" + "5" * 64,
        "expected_cache_fingerprint": "sha256:" + "6" * 64,
        "expected_encoder_artifact_sha256": "sha256:" + "2" * 64,
        "expected_vlm_content_fingerprint": "sha256:" + "7" * 64,
    }


def test_pinned_metadata_rejects_backend_spoof_and_self_consistent_wrong_lineage() -> None:
    _validate_machine_a_metadata(_valid_metadata(), **_pinned_expectation())

    spoof = {"backend": "some-other-backend"}
    with pytest.raises(ValueError, match="does not match configured"):
        _validate_machine_a_metadata(spoof, **_pinned_expectation())

    wrong = _valid_metadata()
    wrong["rl_token_contract"]["checkpoint_fingerprint"] = "sha256:" + "8" * 64
    with pytest.raises(ValueError, match="checkpoint_fingerprint"):
        _validate_machine_a_metadata(wrong, **_pinned_expectation())


def test_generic_legacy_metadata_requires_explicit_unpinned_opt_in() -> None:
    metadata = {"backend": "legacy-openpi"}
    _validate_machine_a_metadata(
        metadata,
        allow_unpinned=True,
        expected_backend=None,
        expected_checkpoint_fingerprint=None,
        expected_cache_fingerprint=None,
        expected_encoder_artifact_sha256=None,
        expected_vlm_content_fingerprint=None,
    )
    with pytest.raises(ValueError, match="requires configured values"):
        _validate_machine_a_metadata(
            metadata,
            allow_unpinned=False,
            expected_backend=None,
            expected_checkpoint_fingerprint=None,
            expected_cache_fingerprint=None,
            expected_encoder_artifact_sha256=None,
            expected_vlm_content_fingerprint=None,
        )
