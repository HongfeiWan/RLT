import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from groot_rlt.integration.artifact_lineage import canonical_json_sha256, file_sha256
from groot_rlt.representation.encoder_artifact import (
    ENCODER_ARTIFACT_KIND,
    FEATURE_TAP,
    PROCESSOR_MODE,
    SOURCE_ARCHITECTURE,
    SOURCE_CHECKPOINT_SCHEMA_VERSION,
    StrictEncoderConfig,
    StrictRLTokenEncoder,
    artifact_manifest_path,
    export_encoder_ema_artifact,
    load_encoder_ema_artifact,
)

CHECKPOINT_FINGERPRINT = "sha256:" + "1" * 64
CACHE_FINGERPRINT = "sha256:" + "2" * 64


def _full_strict_state(config: StrictEncoderConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    encoder = StrictRLTokenEncoder(config)
    state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    state["decoder_query"] = torch.randn(config.max_vl_tokens, config.model_dim)
    state["decoder_memory_pos"] = torch.randn(1, config.model_dim)
    for name, value in list(state.items()):
        if name.startswith("encoder."):
            state[name.replace("encoder.", "decoder.", 1)] = torch.randn_like(value)
    return state


def _write_source_checkpoint(
    path: Path,
    *,
    overrides: dict | None = None,
) -> tuple[StrictEncoderConfig, dict[str, torch.Tensor]]:
    config = StrictEncoderConfig(
        input_dim=8,
        model_dim=8,
        rl_token_dim=8,
        max_vl_tokens=6,
        encoder_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    ema_state = _full_strict_state(config)
    raw_state = {name: value + 0.25 for name, value in ema_state.items()}
    payload = {
        "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
        "architecture": SOURCE_ARCHITECTURE,
        "step": 10_000,
        "autoencoder": ema_state,
        "autoencoder_raw": raw_state,
        "optimizer": {
            "state": {
                name: {
                    "exp_avg": torch.ones_like(value),
                    "exp_avg_sq": torch.ones_like(value),
                }
                for name, value in ema_state.items()
            },
            "param_groups": [{"lr": np.float64(2.5e-6)}],
        },
        "autoencoder_config": {
            **config.__dict__,
            "decoder_layers": config.encoder_layers,
            "use_prefix_mask_token": False,
            "use_decoder_cross_attention": True,
        },
        "args": {
            "representation_lineage": {
                "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
                "cache_fingerprint": CACHE_FINGERPRINT,
                "feature_tap": FEATURE_TAP,
                "processor_mode": PROCESSOR_MODE,
            }
        },
        "ema_decay": 0.99,
        "last_loss": 0.5,
    }
    if overrides:
        payload.update(overrides)
    torch.save(payload, path)
    return config, ema_state


class EncoderArtifactTest(unittest.TestCase):
    def test_encoder_only_forward_matches_training_model_exactly(self) -> None:
        from groot_rlt.representation.train_vl_embedding_autoencoder import (
            RL_TOKEN_ARCHITECTURE,
            RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
            VL_CACHE_FEATURE_TAP,
            VL_CACHE_PROCESSOR_MODE,
            VLTokenAutoencoder,
            VLTokenAutoencoderConfig,
        )

        self.assertEqual(RL_TOKEN_ARCHITECTURE, SOURCE_ARCHITECTURE)
        self.assertEqual(RL_TOKEN_CHECKPOINT_SCHEMA_VERSION, SOURCE_CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(VL_CACHE_FEATURE_TAP, FEATURE_TAP)
        self.assertEqual(VL_CACHE_PROCESSOR_MODE, PROCESSOR_MODE)
        torch.manual_seed(5)
        full_model = VLTokenAutoencoder(
            VLTokenAutoencoderConfig(
                input_dim=8,
                model_dim=8,
                rl_token_dim=8,
                max_vl_tokens=6,
                encoder_layers=2,
                decoder_layers=2,
                num_heads=2,
                mlp_ratio=2.0,
                dropout=0.0,
                use_decoder_cross_attention=True,
            )
        ).eval()
        encoder = StrictRLTokenEncoder(
            StrictEncoderConfig(
                input_dim=8,
                model_dim=8,
                rl_token_dim=8,
                max_vl_tokens=6,
                encoder_layers=2,
                num_heads=2,
                mlp_ratio=2.0,
                dropout=0.0,
            )
        ).eval()
        encoder.load_state_dict(
            {name: full_model.state_dict()[name] for name in encoder.state_dict()}, strict=True
        )
        prefix = torch.randn(2, 6, 8)
        mask = torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, True, True]]
        )
        with torch.no_grad():
            expected = full_model.encode_rl_token(prefix, mask)
            actual = encoder.encode_rl_token(prefix, mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_roundtrip_exports_only_ema_encoder_and_matches_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "010000.pt"
            artifact_path = root / "encoder_ema.pt"
            config, source_ema = _write_source_checkpoint(source_path)

            manifest = export_encoder_ema_artifact(source_path, artifact_path)
            self.assertTrue(artifact_path.is_file())
            self.assertTrue(artifact_manifest_path(artifact_path).is_file())
            self.assertEqual(manifest["artifact_sha256"], f"sha256:{file_sha256(artifact_path)}")
            self.assertLess(artifact_path.stat().st_size, source_path.stat().st_size / 2)

            loaded = load_encoder_ema_artifact(
                artifact_path,
                expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                expected_cache_fingerprint=CACHE_FINGERPRINT,
            )
            self.assertEqual(loaded.metadata["artifact_kind"], ENCODER_ARTIFACT_KIND)
            self.assertFalse(hasattr(loaded.encoder, "decoder"))
            self.assertFalse(
                any(name.startswith("decoder") for name in loaded.encoder.state_dict())
            )
            for name, value in loaded.encoder.state_dict().items():
                torch.testing.assert_close(value, source_ema[name], rtol=0.0, atol=0.0)

            expected = StrictRLTokenEncoder(config).eval()
            expected.load_state_dict(
                {name: source_ema[name] for name in expected.state_dict()}, strict=True
            )
            prefix = torch.randn(2, 5, config.input_dim)
            mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
            with torch.no_grad():
                expected_token = expected.encode_rl_token(prefix, mask)
                actual_token = loaded.encoder.encode_rl_token(prefix, mask)
            torch.testing.assert_close(actual_token, expected_token, rtol=0.0, atol=0.0)

    def test_export_rejects_non_schema2_non_ema_and_conflicting_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "schema": {"schema_version": 1},
                "architecture": {"architecture": "teacher_forced"},
                "not_ema": {"autoencoder_raw": None},
                "missing_lineage": {"args": {}},
            }
            for name, overrides in cases.items():
                with self.subTest(name=name):
                    source = root / f"{name}.pt"
                    output = root / f"{name}_encoder.pt"
                    _write_source_checkpoint(source, overrides=overrides)
                    with self.assertRaises(ValueError):
                        export_encoder_ema_artifact(source, output)

            source = root / "conflicting_lineage.pt"
            _, _ = _write_source_checkpoint(source)
            payload = torch.load(source, map_location="cpu", weights_only=False)
            payload["representation_lineage"] = {
                **payload["args"]["representation_lineage"],
                "cache_fingerprint": "sha256:" + "3" * 64,
            }
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, "conflicting representation lineage"):
                export_encoder_ema_artifact(source, root / "conflicting_encoder.pt")

    def test_export_accepts_equal_explicit_top_level_lineage_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "aliased_source.pt"
            output = root / "encoder.pt"
            _write_source_checkpoint(source)
            payload = torch.load(source, map_location="cpu", weights_only=False)
            payload["representation_lineage"] = dict(payload["args"]["representation_lineage"])
            torch.save(payload, source)
            manifest = export_encoder_ema_artifact(source, output)
            self.assertEqual(
                manifest["representation_lineage"],
                payload["args"]["representation_lineage"],
            )

    def test_loader_rejects_wrong_deployment_lineage_and_file_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            artifact = root / "encoder.pt"
            _write_source_checkpoint(source)
            export_encoder_ema_artifact(source, artifact)

            with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
                load_encoder_ema_artifact(
                    artifact,
                    expected_checkpoint_fingerprint="sha256:" + "4" * 64,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                )
            with self.assertRaisesRegex(ValueError, "cache fingerprint"):
                load_encoder_ema_artifact(
                    artifact,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint="sha256:" + "5" * 64,
                )

            with artifact.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "size|SHA-256"):
                load_encoder_ema_artifact(
                    artifact,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                )

    def test_loader_rejects_tensor_tampering_even_with_updated_file_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            artifact = root / "encoder.pt"
            _write_source_checkpoint(source)
            export_encoder_ema_artifact(source, artifact)

            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            payload["encoder"]["query_token"] = payload["encoder"]["query_token"] + 1.0
            torch.save(payload, artifact)
            manifest_path = artifact_manifest_path(artifact)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_sha256"] = f"sha256:{file_sha256(artifact)}"
            manifest["artifact_size_bytes"] = artifact.stat().st_size
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "tensor-state SHA-256"):
                load_encoder_ema_artifact(
                    artifact,
                    expected_checkpoint_fingerprint=CHECKPOINT_FINGERPRINT,
                    expected_cache_fingerprint=CACHE_FINGERPRINT,
                )

    def test_manifest_metadata_hash_is_bound_to_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            artifact = root / "encoder.pt"
            _write_source_checkpoint(source)
            manifest = export_encoder_ema_artifact(source, artifact)
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            self.assertEqual(
                manifest["metadata_sha256"], canonical_json_sha256(payload["metadata"])
            )


if __name__ == "__main__":
    unittest.main()
