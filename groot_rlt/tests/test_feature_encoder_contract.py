from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from groot_rlt.integration.artifact_lineage import canonical_json_sha256
from groot_rlt.integration.prefix_cache_contract import (
    validate_prefix_cache_deployment_paths,
    vlm_content_fingerprint,
)
from groot_rlt.representation.encoder_artifact import (
    ENCODER_ARCHITECTURE,
    FEATURE_TAP,
    PROCESSOR_MODE,
    SOURCE_ARCHITECTURE,
)
from groot_rlt.serving.groot_feature_policy import (
    FeatureContract,
    PrefixCacheContract,
    _validate_representation_lineage,
    load_prefix_cache_contract,
    load_serving_rl_token_encoder,
)
from groot_rlt.serving.groot_feature_server import make_arg_parser

CHECKPOINT_FINGERPRINT = "sha256:" + "1" * 64
CACHE_FINGERPRINT = "sha256:" + "2" * 64


def _manifest_material(**overrides):
    material = {
        "schema_version": 2,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": FEATURE_TAP,
        "processor_mode": PROCESSOR_MODE,
        "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
        "token_scope": "image",
        "token_sampling": "uniform",
        "max_vl_tokens": 6,
        "input_dim": 8,
        "video_modality_keys": ["ego_view", "wrist_view"],
        "base_model_path": "/model",
        "processor_path": "/model",
        "vlm_model_path": "/vlm",
    }
    material.update(overrides)
    return material


def _write_manifest(path: Path, **overrides) -> str:
    material = _manifest_material(**overrides)
    fingerprint = canonical_json_sha256(material)
    path.write_text(
        json.dumps({**material, "fingerprint": fingerprint}),
        encoding="utf-8",
    )
    return fingerprint


def _contract(path: Path) -> PrefixCacheContract:
    return PrefixCacheContract(
        fingerprint=CACHE_FINGERPRINT,
        checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
        token_scope="image",
        token_sampling="uniform",
        max_vl_tokens=6,
        input_dim=8,
        video_modality_keys=("ego_view", "wrist_view"),
        feature_tap=FEATURE_TAP,
        processor_mode=PROCESSOR_MODE,
        model_path="/model",
        processor_path="/model",
        vlm_model_path="/vlm",
        manifest_path=str(path),
    )


class PrefixCacheContractTest(unittest.TestCase):
    def test_manifest_is_authoritative_and_bound_to_both_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            cache_fingerprint = _write_manifest(path)
            contract = load_prefix_cache_contract(
                path,
                expected_cache_fingerprint=cache_fingerprint,
                expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
            )
            self.assertEqual(contract.token_scope, "image")
            self.assertEqual(contract.token_sampling, "uniform")
            self.assertEqual(contract.max_vl_tokens, 6)
            self.assertEqual(contract.video_modality_keys, ("ego_view", "wrist_view"))

            with self.assertRaisesRegex(ValueError, "expected cache fingerprint"):
                load_prefix_cache_contract(
                    path,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                )
            with self.assertRaisesRegex(ValueError, "400k fingerprint"):
                load_prefix_cache_contract(
                    path,
                    expected_cache_fingerprint=cache_fingerprint,
                    expected_checkpoint_fingerprint="sha256:" + "3" * 64,
                )

    def test_manifest_tampering_and_random_sampling_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            fingerprint = _write_manifest(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["max_vl_tokens"] = 7
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint is invalid"):
                load_prefix_cache_contract(
                    path,
                    expected_cache_fingerprint=fingerprint,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                )

            random_fingerprint = _write_manifest(path, token_sampling="random")
            with self.assertRaisesRegex(ValueError, "deterministic"):
                load_prefix_cache_contract(
                    path,
                    expected_cache_fingerprint=random_fingerprint,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                )

    def test_wrong_processor_or_vlm_path_fails_before_model_loading(self) -> None:
        contract = _contract(Path("/cache/manifest.json"))
        with self.assertRaisesRegex(ValueError, "processor_path"):
            validate_prefix_cache_deployment_paths(
                contract,
                model_path="/model",
                processor_path="/other-processor",
                vlm_model_path="/vlm",
                context="Serving",
            )
        with self.assertRaisesRegex(ValueError, "vlm_model_path"):
            validate_prefix_cache_deployment_paths(
                contract,
                model_path="/model",
                processor_path="/model",
                vlm_model_path="/other-vlm",
                context="Offline",
            )

    def test_vlm_content_fingerprint_changes_with_deployment_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            weights = root / "model.safetensors"
            weights.write_bytes(b"weights-v1")
            first = vlm_content_fingerprint(root)
            self.assertEqual(first, vlm_content_fingerprint(root))
            weights.write_bytes(b"weights-v2")
            self.assertNotEqual(first, vlm_content_fingerprint(root))


class ServingEncoderSourceTest(unittest.TestCase):
    def test_encoder_artifact_is_explicit_and_advertises_verified_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "encoder.pt"
            artifact.touch()
            manifest = artifact.with_suffix(".pt.manifest.json")
            manifest_payload = {
                "metadata_sha256": "sha256:" + "3" * 64,
                "artifact_sha256": "sha256:" + "4" * 64,
                "encoder_state_sha256": "sha256:" + "5" * 64,
                "source_checkpoint_sha256": "sha256:" + "6" * 64,
            }
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            encoder = SimpleNamespace(
                config=SimpleNamespace(input_dim=8, max_vl_tokens=6, rl_token_dim=8)
            )
            loaded = SimpleNamespace(
                encoder=encoder,
                artifact_path=artifact,
                manifest_path=manifest,
            )
            with mock.patch(
                "groot_rlt.serving.groot_feature_policy.load_encoder_ema_artifact",
                return_value=loaded,
            ) as artifact_loader:
                result = load_serving_rl_token_encoder(
                    encoder_artifact=artifact,
                    legacy_full_checkpoint=None,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                    cache_contract=_contract(root / "cache.json"),
                    device="cpu",
                )
            artifact_loader.assert_called_once()
            self.assertIs(result.encoder, encoder)
            self.assertEqual(result.handshake["architecture"], ENCODER_ARCHITECTURE)
            self.assertEqual(result.handshake["artifact_file_sha256"], "sha256:" + "4" * 64)
            self.assertEqual(result.handshake["encoder_tensor_sha256"], "sha256:" + "5" * 64)
            self.assertEqual(result.handshake["checkpoint_fingerprint"], CHECKPOINT_FINGERPRINT)
            self.assertEqual(result.handshake["cache_fingerprint"], CACHE_FINGERPRINT)
            self.assertEqual(result.handshake["representation_checkpoint_schema_version"], 2)
            self.assertEqual(result.handshake["prefix_cache_schema_version"], 2)

    def test_source_selection_never_falls_back(self) -> None:
        contract = _contract(Path("/cache/manifest.json"))
        for artifact, legacy in ((None, None), ("bad.pt", "legacy.pt")):
            with self.subTest(artifact=artifact, legacy=legacy):
                with self.assertRaisesRegex(ValueError, "Exactly one"):
                    load_serving_rl_token_encoder(
                        encoder_artifact=artifact,
                        legacy_full_checkpoint=legacy,
                        expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                        expected_cache_fingerprint=CACHE_FINGERPRINT,
                        cache_contract=contract,
                        device="cpu",
                    )
        with mock.patch(
            "groot_rlt.serving.groot_feature_policy.load_encoder_ema_artifact",
            side_effect=ValueError("corrupt artifact"),
        ):
            with self.assertRaisesRegex(ValueError, "corrupt artifact"):
                load_serving_rl_token_encoder(
                    encoder_artifact="bad.pt",
                    legacy_full_checkpoint=None,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                    cache_contract=contract,
                    device="cpu",
                )

    def test_artifact_shape_and_legacy_lineage_mismatch_fail_closed(self) -> None:
        encoder = SimpleNamespace(
            config=SimpleNamespace(input_dim=9, max_vl_tokens=6, rl_token_dim=8)
        )
        with mock.patch(
            "groot_rlt.serving.groot_feature_policy.load_encoder_ema_artifact",
            return_value=SimpleNamespace(encoder=encoder),
        ):
            with self.assertRaisesRegex(ValueError, "input_dim"):
                load_serving_rl_token_encoder(
                    encoder_artifact="encoder.pt",
                    legacy_full_checkpoint=None,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                    cache_contract=_contract(Path("/cache/manifest.json")),
                    device="cpu",
                )
        with self.assertRaisesRegex(ValueError, "lacks args.representation_lineage"):
            _validate_representation_lineage(
                {},
                expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                expected_cache_fingerprint=CACHE_FINGERPRINT,
            )

    def test_explicit_legacy_full_checkpoint_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "strict.pt"
            checkpoint.write_bytes(b"trusted legacy checkpoint")
            encoder = SimpleNamespace(
                config=SimpleNamespace(input_dim=8, max_vl_tokens=6, rl_token_dim=8),
                state_dict=lambda: {"query_token": torch.zeros(1, 8)},
            )
            args = {
                "representation_lineage": {
                    "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
                    "cache_fingerprint": CACHE_FINGERPRINT,
                    "feature_tap": FEATURE_TAP,
                    "processor_mode": PROCESSOR_MODE,
                },
                "token_scope": "image",
                "token_sampling": "uniform",
                "max_vl_tokens": 6,
            }
            with mock.patch(
                "groot_rlt.representation.visualize_rl_token_umap.load_rl_token_encoder",
                return_value=(encoder, args, 10_000, 0.1),
            ):
                result = load_serving_rl_token_encoder(
                    encoder_artifact=None,
                    legacy_full_checkpoint=checkpoint,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                    cache_contract=_contract(Path(temporary) / "cache.json"),
                    device="cpu",
                )
            self.assertEqual(result.handshake["source_kind"], "legacy_full_checkpoint")
            self.assertEqual(result.handshake["architecture"], SOURCE_ARCHITECTURE)


class FeatureServerArgumentTest(unittest.TestCase):
    def _base_args(self) -> list[str]:
        return [
            "--model-path",
            "/model",
            "--vlm-model-path",
            "/vlm",
            "--prefix-cache-manifest",
            "/cache/manifest.json",
            "--expected-checkpoint-fingerprint",
            CHECKPOINT_FINGERPRINT,
            "--expected-cache-fingerprint",
            CACHE_FINGERPRINT,
            "--expected-vlm-content-fingerprint",
            "sha256:" + "7" * 64,
        ]

    def test_artifact_is_primary_and_legacy_source_is_explicit(self) -> None:
        parser = make_arg_parser()
        artifact_args = parser.parse_args(
            [*self._base_args(), "--rl-token-encoder-artifact", "/encoder.pt"]
        )
        self.assertEqual(artifact_args.rl_token_encoder_artifact, "/encoder.pt")
        self.assertIsNone(artifact_args.legacy_rl_token_checkpoint)
        legacy_args = parser.parse_args(
            [*self._base_args(), "--legacy-rl-token-checkpoint", "/full.pt"]
        )
        self.assertEqual(legacy_args.legacy_rl_token_checkpoint, "/full.pt")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    *self._base_args(),
                    "--rl-token-encoder-artifact",
                    "/encoder.pt",
                    "--legacy-rl-token-checkpoint",
                    "/full.pt",
                ]
            )

    def test_nero_proprio_default_matches_feature_contract(self) -> None:
        parser = make_arg_parser()
        args = parser.parse_args(
            [*self._base_args(), "--rl-token-encoder-artifact", "/encoder.pt"]
        )

        self.assertEqual(args.proprio_dim, 19)
        self.assertEqual(FeatureContract().proprio_dim, 19)


if __name__ == "__main__":
    unittest.main()
