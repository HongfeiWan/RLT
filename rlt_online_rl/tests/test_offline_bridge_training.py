# ruff: noqa: E402

from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import ActorServiceConfig
from rlt_online_rl.config import LearnerServiceConfig
from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.inference import ActorService
from rlt_online_rl.offline_bridge_training import BRIDGE_SCHEMA_NAME
from rlt_online_rl.offline_bridge_training import BRIDGE_SCHEMA_VERSION
from rlt_online_rl.offline_bridge_training import PROPRIO_LAYOUT
from rlt_online_rl.offline_bridge_training import PROPRIO_LAYOUT_HASH
from rlt_online_rl.offline_bridge_training import OfflineBridgeTrainConfig
from rlt_online_rl.offline_bridge_training import load_and_validate_bridge_bundle
from rlt_online_rl.offline_bridge_training import run_offline_bridge_training
from rlt_online_rl.replay import ReplayManager
from rlt_online_rl.replay import ReplayTensorContract

DATASET_FINGERPRINT = f"sha256:{'1' * 64}"
FEATURE_FINGERPRINT = f"sha256:{'2' * 64}"
BRIDGE_FINGERPRINT = f"sha256:{'3' * 64}"


def _synthetic_bundle(*, step_count: int = 14, z_dim: int = 4) -> dict:
    segment_id = "synthetic:episode_000000:run_0000:policy_rollout:frames_000000_000013"
    steps = []
    for index in range(step_count):
        done = index == step_count - 1
        next_index = index if done else index + 1
        steps.append(
            {
                "z_rl": np.full((z_dim,), index + 0.25, dtype=np.float32),
                "proprio": np.full((19,), index + 0.5, dtype=np.float32),
                "vla_reference_action": np.concatenate(
                    (
                        np.full((19,), index + 0.75, dtype=np.float32),
                        np.full((7,), index + 10.75, dtype=np.float32),
                    )
                ),
                "ref_action": np.full((19,), index + 0.75, dtype=np.float32),
                "action": np.full((19,), index + 1.0, dtype=np.float32),
                "reward": 1.0 if done else 0.0,
                "done": done,
                "next_z_rl": np.full((z_dim,), next_index + 0.25, dtype=np.float32),
                "next_proprio": np.full((19,), next_index + 0.5, dtype=np.float32),
                "source": 0,
                "collection_phase": "warmup",
                "success": int(done),
                "intervention_flag": False,
                "episode_id": 0,
                "step_id": index,
                "dataset_fingerprint": DATASET_FINGERPRINT,
                "feature_contract_fingerprint": FEATURE_FINGERPRINT,
                "bridge_fingerprint": BRIDGE_FINGERPRINT,
                "task_episode_id": "synthetic-task-0",
                "source_episode_index": 0,
                "source_frame_index": index,
                "next_source_frame_index": next_index,
                "source_partition": "policy_rollout",
                "segment_id": segment_id,
                "source_segment_id": segment_id,
            }
        )
    manifest = {
        "schema_name": BRIDGE_SCHEMA_NAME,
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "source_format": "lerobot_v3_dagger",
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "feature_contract_fingerprint": FEATURE_FINGERPRINT,
        "bridge_fingerprint": BRIDGE_FINGERPRINT,
        "fps": 10.0,
        "state_contract": {
            "source_dim": 26,
            "runtime_dim": 19,
            "rotation_convention": "groot_row_major_first_two_rows",
            "source_order": "arm7+eef9+hand10",
            "runtime_order": "eef9+hand10",
            "projection": list(range(7, 26)),
            "runtime_layout": list(PROPRIO_LAYOUT),
            "runtime_layout_hash": PROPRIO_LAYOUT_HASH,
            "rotation_transposed": False,
        },
        "action_contract": {
            "dimension": 19,
            "semantics": "post_guard_safe_action_in_policy_state_frame",
            "rotation_convention": "groot_row_major_first_two_rows",
        },
        "reference_contract": {
            "source_dimension": 26,
            "runtime_dimension": 19,
            "projection": list(range(19)),
            "source_order": "eef9+hand10+arm7",
            "runtime_order": "eef9+hand10",
            "source_field": "vla_reference_action",
            "runtime_field": "ref_action",
            "learner_consumes_source": False,
        },
        "source_frame_count": step_count,
        "replay_step_count": step_count,
        "segment_count": 1,
        "dropped_boundary_frame_count": 0,
        "outcome_counts": {"success": 1, "failure": 0},
        "partition_boundary_policy": "split_and_drop_nonterminal_tail",
        "terminal_next_state_policy": "self_state_bootstrap_masked_by_done",
        "recommended_allow_partial": False,
    }
    return {
        "schema_name": BRIDGE_SCHEMA_NAME,
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "manifest": manifest,
        "episodes": [
            {
                "segment_id": segment_id,
                "dataset_fingerprint": DATASET_FINGERPRINT,
                "feature_contract_fingerprint": FEATURE_FINGERPRINT,
                "bridge_fingerprint": BRIDGE_FINGERPRINT,
                "source_episode_index": 0,
                "task_episode_id": "synthetic-task-0",
                "partition": "policy_rollout",
                "source_frame_start": 0,
                "source_frame_end_inclusive": step_count - 1,
                "dropped_tail_frame_index": None,
                "steps": steps,
            }
        ],
    }


def _write_bundle(path: Path, payload: dict) -> None:
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _production_rl_config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=19,
        chunk_len=10,
        z_dim=4,
        proprio_dim=19,
        rot6d_convention="groot_row_major_first_two_rows",
        delta_action_indices=(),
        actor_hidden_dim=16,
        actor_num_layers=1,
        critic_hidden_dim=16,
        critic_num_layers=1,
        warmup_min_size=1,
    )


def test_synthetic_bridge_builds_replay_and_runs_nonproduction_smoke(tmp_path) -> None:
    bridge_path = tmp_path / "bridge.pkl"
    output_dir = tmp_path / "offline-smoke"
    _write_bundle(bridge_path, _synthetic_bundle())

    summary = run_offline_bridge_training(
        bridge_path,
        output_dir,
        config=OfflineBridgeTrainConfig(
            train_steps=2,
            batch_size=2,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
        ),
    )

    assert summary["status"] == "ok"
    assert summary["artifact_mode"] == "non_production_smoke"
    assert not summary["production_ready"]
    assert not summary["features_are_real"]
    assert summary["feature_contract"]["fingerprint"] == FEATURE_FINGERPRINT
    assert not summary["feature_contract"]["explicitly_verified"]
    assert summary["contracts"]["state_source_dim"] == 26
    assert summary["contracts"]["proprio_dim"] == 19
    assert summary["contracts"]["action_dim"] == 19
    assert summary["contracts"]["chunk_len"] == 10
    assert summary["contracts"]["stride"] == 2
    assert summary["replay"]["transition_count"] == 3
    assert summary["training"]["completed_steps"] == 2
    assert summary["training"]["actor_version"] == 1

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["features_are_real"] is False
    assert metrics["training"]["completed_steps"] == 2
    with (output_dir / "actor_snapshot.pkl").open("rb") as stream:
        actor_snapshot = pickle.load(stream)
    assert actor_snapshot["bridge_lineage"]["artifact_mode"] == "non_production_smoke"
    assert actor_snapshot["bridge_lineage"]["features_are_real"] is False
    assert actor_snapshot["rl_config"]["action_dim"] == 19
    assert actor_snapshot["rl_config"]["proprio_dim"] == 19
    with (output_dir / "learner_checkpoint.pkl").open("rb") as stream:
        learner_checkpoint = pickle.load(stream)
    assert learner_checkpoint["state"]["global_step"] == 2
    assert learner_checkpoint["bridge_lineage"]["feature_contract"]["fingerprint"] == FEATURE_FINGERPRINT
    assert not list(output_dir.glob("*.tmp"))

    restored = ReplayManager(
        32,
        journal_path=str(output_dir / "replay_journal.pkl"),
        tensor_contract=ReplayTensorContract(z_dim=4, proprio_dim=19, chunk_len=10, action_dim=19),
    )
    assert restored.stats()["size"] == 3


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda payload: payload["manifest"]["state_contract"].update(runtime_dim=18), "runtime_dim"),
        (lambda payload: payload.update(schema_version=1), "schema_version"),
        (
            lambda payload: payload["episodes"][0]["steps"][0].update(
                vla_reference_action=np.zeros((25,), dtype=np.float32)
            ),
            "vla_reference_action",
        ),
        (
            lambda payload: payload["episodes"][0]["steps"][0].update(
                ref_action=np.ones((19,), dtype=np.float32)
            ),
            "exactly equal",
        ),
        (lambda payload: payload["manifest"]["action_contract"].update(dimension=18), "dimension"),
        (lambda payload: payload["manifest"].update(fps=15.0), "fps"),
        (
            lambda payload: payload["manifest"].update(feature_contract_fingerprint="not-a-fingerprint"),
            "fingerprint",
        ),
        (lambda payload: payload["manifest"].update(segment_count=2), "segment_count"),
        (
            lambda payload: payload["episodes"][0]["steps"][0].update(segment_id="wrong-segment"),
            "segment_id",
        ),
    ],
)
def test_bridge_manifest_and_segment_contracts_fail_closed(tmp_path, mutation, error_match) -> None:
    payload = copy.deepcopy(_synthetic_bundle())
    mutation(payload)
    bridge_path = tmp_path / "bad.pkl"
    _write_bundle(bridge_path, payload)

    with pytest.raises((TypeError, ValueError), match=error_match):
        load_and_validate_bridge_bundle(bridge_path)


def test_real_feature_and_production_labels_require_explicit_contract(tmp_path) -> None:
    with pytest.raises(ValueError, match="features_are_real requires"):
        OfflineBridgeTrainConfig(features_are_real=True)
    with pytest.raises(ValueError, match="production_run requires"):
        OfflineBridgeTrainConfig(production_run=True)

    bridge_path = tmp_path / "bridge.pkl"
    _write_bundle(bridge_path, _synthetic_bundle())
    with pytest.raises(ValueError, match="feature contract fingerprint mismatch"):
        load_and_validate_bridge_bundle(
            bridge_path,
            expected_feature_contract_fingerprint=f"sha256:{'9' * 64}",
        )

    config = OfflineBridgeTrainConfig(
        features_are_real=True,
        expected_feature_contract_fingerprint=FEATURE_FINGERPRINT,
        production_run=True,
    )
    with pytest.raises(ValueError, match="explicit production rl_config"):
        run_offline_bridge_training(bridge_path, tmp_path / "production", config=config)
    with pytest.raises(ValueError, match="full explicit online RL system config"):
        run_offline_bridge_training(
            bridge_path,
            tmp_path / "production",
            config=config,
            rl_config=_production_rl_config(),
        )


def test_verified_real_feature_run_can_emit_production_lineage(tmp_path) -> None:
    bridge_path = tmp_path / "bridge.pkl"
    output_dir = tmp_path / "production"
    _write_bundle(bridge_path, _synthetic_bundle())
    config = OfflineBridgeTrainConfig(
        train_steps=1,
        batch_size=2,
        features_are_real=True,
        expected_feature_contract_fingerprint=FEATURE_FINGERPRINT,
        production_run=True,
    )
    rl_config = _production_rl_config()

    summary = run_offline_bridge_training(
        bridge_path,
        output_dir,
        config=config,
        rl_config=rl_config,
        system_config=OnlineRLSystemConfig(rl=rl_config),
    )

    assert summary["artifact_mode"] == "production"
    assert summary["production_ready"]
    assert summary["features_are_real"]
    assert summary["feature_contract"]["explicitly_verified"]
    assert summary["feature_contract"]["expected_fingerprint"] == FEATURE_FINGERPRINT
    actor_path = output_dir / "actor_snapshot" / "actor_snapshot.pkl"
    checkpoint_dir = output_dir / "checkpoints"
    replay_path = output_dir / "replay" / "replay_journal.pkl"
    assert (checkpoint_dir / "latest.pkl").is_file()
    assert (checkpoint_dir / "online_rl_config.yaml").is_file()
    assert replay_path.is_file()
    assert (output_dir / "manifest.json").is_file()
    resolved_system = load_system_config_yaml(str(checkpoint_dir / "online_rl_config.yaml"))
    assert Path(resolved_system.actor_service.snapshot_path) == actor_path
    assert Path(resolved_system.learner_service.checkpoint_dir) == checkpoint_dir
    assert Path(resolved_system.learner_service.actor_snapshot_path) == actor_path
    assert Path(resolved_system.replay.journal_path) == replay_path
    with actor_path.open("rb") as stream:
        actor_snapshot = pickle.load(stream)
    assert actor_snapshot["bridge_lineage"]["production_ready"]

    class _ReplaySource:
        def stats(self):
            return {
                "size": 3,
                "adds_total": 3,
                "max_episode_id": 0,
                "recent_episode_window": 20,
            }

        def sample_batch(self, _batch_size):  # pragma: no cover - restore does not train.
            raise AssertionError("restore test must not sample replay")

    from rlt_online_rl.trainer import LearnerService

    learner = LearnerService(
        rl_config,
        LearnerServiceConfig(
            checkpoint_dir=str(checkpoint_dir),
            actor_snapshot_path=str(actor_path),
        ),
        _ReplaySource(),
    )
    assert int(learner.state.global_step) == 1

    actor_service = ActorService(
        rl_config,
        ActorServiceConfig(snapshot_path=str(actor_path), pull_params_interval_sec=60.0),
    )
    try:
        assert actor_service.actor_param_version == int(learner.state.actor_version)
    finally:
        actor_service._stop_event.set()
        if actor_service._poll_thread is not None:
            actor_service._poll_thread.join(timeout=1.0)
