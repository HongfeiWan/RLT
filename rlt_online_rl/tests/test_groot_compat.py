from __future__ import annotations

# The standalone tests support a source checkout without requiring an editable install.
# ruff: noqa: E402
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.inference import MachineAFeatureClient
from rlt_online_rl.inference import normalize_feature_payload


def _layout_hash(names: list[str], rotation_convention: str | None) -> str:
    encoded = json.dumps(
        {
            "channel_names": names,
            "rotation_convention": rotation_convention,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _write_stats(path: Path, action_dim: int) -> None:
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {
                        "q01": [-2.0] * action_dim,
                        "q99": [2.0] * action_dim,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_nero_26d_noncontiguous_delta_indices_roundtrip(tmp_path) -> None:
    stats_path = tmp_path / "stats.json"
    _write_stats(stats_path, 26)
    cfg = RLTOnlineRLConfig(
        action_dim=26,
        proprio_dim=26,
        action_representation="delta_chunk",
        action_norm_stats_path=str(stats_path),
        delta_action_indices=(0, 8, 25),
    )
    adapter = ActionRepresentationAdapter.from_config(cfg)
    assert adapter is not None

    state = np.linspace(-0.5, 0.5, 26, dtype=np.float32)
    absolute = np.stack([state + 0.01 * step for step in range(10)], axis=0)
    normalized = adapter.normalize_chunk(absolute, state)
    represented = absolute.copy()
    represented[:, [0, 8, 25]] -= state[[0, 8, 25]]
    expected_normalized = (represented + 2.0) / (4.0 + 1e-6) * 2.0 - 1.0
    assert np.allclose(normalized, expected_normalized, atol=2e-6)
    restored = adapter.denormalize_to_abs_chunk(normalized, state)
    assert np.allclose(restored, absolute, atol=2e-6)


def test_machine_a_payload_proprio_supports_nested_groot_observation() -> None:
    cfg = RLTOnlineRLConfig(action_dim=26, proprio_dim=26, chunk_len=10, z_dim=2048)
    payload = {
        "z_rl": np.zeros((2048,), dtype=np.float32),
        "ref_chunk": np.zeros((10, 26), dtype=np.float32),
        "proprio": np.arange(26, dtype=np.float32),
    }
    observation = {"state": {"eef_hand_arm": np.zeros((1, 1, 26), dtype=np.float32)}}
    normalized = normalize_feature_payload(payload, cfg, observation=observation)
    assert np.array_equal(normalized["proprio"], payload["proprio"])


def test_flat_legacy_observation_remains_authoritative_for_proprio() -> None:
    cfg = RLTOnlineRLConfig(action_dim=7, proprio_dim=7, chunk_len=10, z_dim=8)
    payload = {
        "z_rl": np.zeros((8,), dtype=np.float32),
        "ref_chunk": np.zeros((10, 7), dtype=np.float32),
        "proprio": np.full((7,), 99.0, dtype=np.float32),
    }
    observation = {"state": np.arange(7, dtype=np.float32)}
    normalized = normalize_feature_payload(payload, cfg, observation=observation)
    assert np.array_equal(normalized["proprio"], observation["state"])


def test_layout_validated_flat_proprio_rejects_implicit_truncation() -> None:
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        proprio_dim=3,
        chunk_len=2,
        z_dim=4,
        action_layout_hash="sha256:action",
        proprio_layout_hash="sha256:proprio",
    )
    payload = {
        "z_rl": np.zeros((4,), dtype=np.float32),
        "ref_chunk": np.zeros((2, 3), dtype=np.float32),
        "action_layout_hash": "sha256:action",
        "proprio_layout_hash": "sha256:proprio",
    }

    with pytest.raises(ValueError, match="requires an explicit payload proprio"):
        normalize_feature_payload(
            payload,
            cfg,
            observation={"state": np.zeros((4,), dtype=np.float32)},
        )


def test_machine_a_layout_mismatch_fails_before_action_use() -> None:
    proprio_layout = ["state[0]", "state[1]", "state[2]"]
    proprio_hash = _layout_hash(proprio_layout, None)
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        proprio_dim=3,
        chunk_len=2,
        z_dim=4,
        action_layout_hash="sha256:expected-action",
        proprio_layout_hash=proprio_hash,
    )
    payload = {
        "z_rl": np.zeros((4,), dtype=np.float32),
        "ref_chunk": np.zeros((2, 3), dtype=np.float32),
        "proprio": np.zeros((3,), dtype=np.float32),
        "action_layout_hash": "sha256:wrong",
        "proprio_layout_hash": proprio_hash,
        "proprio_layout": proprio_layout,
    }
    with pytest.raises(ValueError, match="action_layout_hash"):
        normalize_feature_payload(
            payload,
            cfg,
            observation={"state": np.zeros((3,), dtype=np.float32)},
        )


def test_nero_pinned_payload_uses_19d_proprio_and_separate_26d_reference_projection() -> None:
    rot6d_convention = "groot_row_major_first_two_rows"
    proprio_layout = [
        *(f"eef_9d[{index}]" for index in range(9)),
        *(f"hand_joint_pos[{index}]" for index in range(10)),
    ]
    source_action_layout = [f"checkpoint_action[{index}]" for index in range(26)]
    target_action_layout = source_action_layout[:19]
    proprio_hash = _layout_hash(proprio_layout, rot6d_convention)
    source_action_hash = _layout_hash(source_action_layout, rot6d_convention)
    target_action_hash = _layout_hash(target_action_layout, rot6d_convention)
    cfg = RLTOnlineRLConfig(
        action_dim=19,
        proprio_dim=19,
        chunk_len=2,
        z_dim=4,
        action_layout_hash=target_action_hash,
        proprio_layout_hash=proprio_hash,
        reference_action_dim=26,
        reference_action_layout_hash=source_action_hash,
        reference_action_indices=tuple(range(19)),
        rot6d_convention=rot6d_convention,
    )
    full_reference = np.arange(52, dtype=np.float32).reshape(2, 26)
    payload_proprio = np.arange(100, 119, dtype=np.float32)
    payload = {
        "z_rl": np.zeros((4,), dtype=np.float32),
        "ref_chunk": full_reference,
        "proprio": payload_proprio,
        "action_layout_hash": source_action_hash,
        "action_layout": source_action_layout,
        "proprio_layout_hash": proprio_hash,
        "proprio_layout": proprio_layout,
        "rot6d_convention": rot6d_convention,
    }

    normalized = normalize_feature_payload(
        payload,
        cfg,
        observation={"state": np.arange(26, dtype=np.float32)},
    )

    assert np.array_equal(normalized["proprio"], payload_proprio)
    assert not np.array_equal(normalized["proprio"], np.arange(19, dtype=np.float32))
    assert np.array_equal(normalized["ref_chunk"], full_reference[:, :19])
    assert np.array_equal(normalized["source_ref_chunk"], full_reference)


def test_nero_pinned_payload_rejects_layout_names_that_do_not_match_hash() -> None:
    layout = [f"state[{index}]" for index in range(19)]
    expected_hash = _layout_hash(layout, None)
    payload = {
        "z_rl": np.zeros((4,), dtype=np.float32),
        "ref_chunk": np.zeros((2, 19), dtype=np.float32),
        "proprio": np.zeros((19,), dtype=np.float32),
        "proprio_layout_hash": expected_hash,
        "proprio_layout": list(reversed(layout)),
    }
    cfg = RLTOnlineRLConfig(
        action_dim=19,
        proprio_dim=19,
        chunk_len=2,
        z_dim=4,
        proprio_layout_hash=expected_hash,
    )

    with pytest.raises(ValueError, match="proprio layout hash"):
        normalize_feature_payload(
            payload,
            cfg,
            observation={"state": np.arange(26, dtype=np.float32)},
        )


def test_yaml_coerces_delta_action_indices_to_hashable_tuple(tmp_path) -> None:
    path = tmp_path / "online_rl.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "rl": {
                        "action_dim": 26,
                        "proprio_dim": 26,
                        "delta_action_indices": [0, 8, 25],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = load_system_config_yaml(str(path))
    assert config.rl.delta_action_indices == (0, 8, 25)
    assert isinstance(hash(config.rl), int)


def test_symmetric_horizon_stats_use_declared_minmax_bounds(tmp_path) -> None:
    stats_path = tmp_path / "symmetric_stats.json"
    lower = [[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]]
    upper = [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    stats_path.write_text(
        json.dumps(
            {
                "normalization": {
                    "action_representation": "abs",
                    "mode": "symmetric_quantile",
                    "lower_key": "min",
                    "upper_key": "max",
                },
                "layout": {"layout_hash": "sha256:test"},
                "norm_stats": {
                    "actions": {
                        "q01": [[-1.0] * 3] * 2,
                        "q99": [[1.0] * 3] * 2,
                        "min": lower,
                        "max": upper,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        proprio_dim=3,
        chunk_len=2,
        action_representation="abs",
        action_norm_stats_path=str(stats_path),
        action_layout_hash="sha256:test",
    )
    adapter = ActionRepresentationAdapter.from_config(cfg)
    assert adapter is not None
    assert np.array_equal(adapter.stats.q01, np.asarray(lower, dtype=np.float32))
    chunk = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    normalized = adapter.normalize_chunk(chunk, np.zeros((3,), dtype=np.float32))
    restored = adapter.denormalize_to_abs_chunk(normalized, np.zeros((3,), dtype=np.float32))
    assert np.allclose(restored, chunk, atol=2e-6)


def test_stats_layout_mismatch_fails_closed(tmp_path) -> None:
    stats_path = tmp_path / "stats.json"
    _write_stats(stats_path, 3)
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        proprio_dim=3,
        action_norm_stats_path=str(stats_path),
        action_layout_hash="sha256:expected",
    )
    with pytest.raises(ValueError, match="layout hash"):
        ActionRepresentationAdapter.from_config(cfg)


def test_machine_a_client_honors_server_batch_capability(monkeypatch) -> None:
    client = object.__new__(MachineAFeatureClient)
    client._metadata = {"supports_batch": False}  # noqa: SLF001
    calls = []

    def fake_get_features(observation):
        calls.append(observation)
        return {"id": observation["id"]}

    monkeypatch.setattr(client, "get_features", fake_get_features)
    observations = [{"id": 1}, {"id": 2}]
    assert client.get_features_batch(observations) == [{"id": 1}, {"id": 2}]
    assert calls == observations
