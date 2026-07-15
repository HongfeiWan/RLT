from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import relativize_rl_config_paths
from rlt_online_rl.config import save_system_config_yaml
from rlt_online_rl.replay import ReplayManager
from rlt_online_rl.replay import ReplayTensorContract
from rlt_online_rl.replay import RLTTransition
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.replay import build_chunk_transitions_from_episode
from rlt_online_rl.replay import build_terminal_aligned_chunk_transition
from rlt_online_rl.trainer import RLTTrainState
from rlt_online_rl.trainer import init_train_state
from rlt_online_rl.trainer import prepare_replay_training_batch
from rlt_online_rl.trainer import train_step

BRIDGE_SCHEMA_NAME = "groot_rlt.lerobot_v3_dagger_replay"
BRIDGE_SCHEMA_VERSION = 2
SOURCE_FORMAT = "lerobot_v3_dagger"
STATE_SOURCE_DIM = 26
PROPRIO_DIM = 19
ACTION_DIM = 19
REFERENCE_SOURCE_DIM = 26
CHUNK_LEN = 10
STRIDE = 2
ROT6D_CONVENTION = "groot_row_major_first_two_rows"
STATE_TO_PROPRIO_INDICES = tuple(range(7, 26))
REFERENCE_TO_ACTION_INDICES = tuple(range(19))
PROPRIO_LAYOUT = (
    *(f"eef_9d[{index}]" for index in range(9)),
    *(f"hand_joint_pos[{index}]" for index in range(10)),
)
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PARTITION_SOURCE = {
    "policy_rollout": int(TransitionSource.BASE),
    "human_correction": int(TransitionSource.HUMAN),
    "human_demo": int(TransitionSource.HUMAN),
}


def _semantic_layout_hash(channel_names: Sequence[str], rotation_convention: str) -> str:
    payload = json.dumps(
        {
            "channel_names": list(channel_names),
            "rotation_convention": rotation_convention,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


PROPRIO_LAYOUT_HASH = _semantic_layout_hash(PROPRIO_LAYOUT, ROT6D_CONVENTION)


@dataclasses.dataclass(frozen=True, slots=True)
class OfflineBridgeTrainConfig:
    """Controls a bounded offline replay build and JAX training smoke run."""

    train_steps: int = 2
    batch_size: int = 8
    seed: int = 0
    expected_fps: float = 10.0
    replay_capacity: int = 200_000
    metrics_interval: int = 1
    actor_hidden_dim: int = 64
    actor_num_layers: int = 1
    critic_hidden_dim: int = 64
    critic_num_layers: int = 1
    bc_weight: float | None = None
    q_weight: float | None = None
    delta_weight: float | None = None
    features_are_real: bool = False
    expected_feature_contract_fingerprint: str | None = None
    production_run: bool = False

    def __post_init__(self) -> None:
        if self.train_steps < 0:
            raise ValueError("train_steps must be non-negative")
        for name in (
            "batch_size",
            "replay_capacity",
            "metrics_interval",
            "actor_hidden_dim",
            "actor_num_layers",
            "critic_hidden_dim",
            "critic_num_layers",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.expected_fps) or self.expected_fps <= 0.0:
            raise ValueError("expected_fps must be finite and positive")
        if self.expected_feature_contract_fingerprint is not None:
            _require_fingerprint(
                "expected_feature_contract_fingerprint",
                self.expected_feature_contract_fingerprint,
            )
        if self.features_are_real and self.expected_feature_contract_fingerprint is None:
            raise ValueError(
                "features_are_real requires expected_feature_contract_fingerprint so a fake "
                "provider bundle cannot be promoted implicitly"
            )
        if self.production_run and not self.features_are_real:
            raise ValueError("production_run requires features_are_real=true")
        if self.production_run and self.train_steps == 0:
            raise ValueError("production_run requires at least one train step")
        for name in ("bc_weight", "q_weight", "delta_weight"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedBridgeBundle:
    manifest: dict[str, Any]
    episodes: tuple[dict[str, Any], ...]
    z_dim: int
    dataset_fingerprint: str
    feature_contract_fingerprint: str
    bridge_fingerprint: str


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_sequence(name: str, value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence, got {type(value).__name__}")
    return value


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _require_fingerprint(name: str, value: Any) -> str:
    result = str(value)
    if _FINGERPRINT_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{name} must match sha256:<64 lowercase hex>, got {result!r}")
    return result


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _finite_array(name: str, value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_manifest_contract(manifest: Mapping[str, Any], *, expected_fps: float) -> None:
    _require_equal("manifest.schema_name", manifest.get("schema_name"), BRIDGE_SCHEMA_NAME)
    _require_equal("manifest.schema_version", manifest.get("schema_version"), BRIDGE_SCHEMA_VERSION)
    _require_equal("manifest.source_format", manifest.get("source_format"), SOURCE_FORMAT)

    fps_value = manifest.get("fps")
    if isinstance(fps_value, bool | np.bool_) or not isinstance(fps_value, int | float | np.number):
        raise TypeError("manifest.fps must be numeric")
    fps = float(fps_value)
    if not math.isfinite(fps) or not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"manifest.fps must be exactly {expected_fps}, got {fps}")

    state = _require_mapping("manifest.state_contract", manifest.get("state_contract"))
    _require_equal(
        "manifest.state_contract.source_dim",
        state.get("source_dim"),
        STATE_SOURCE_DIM,
    )
    _require_equal("manifest.state_contract.runtime_dim", state.get("runtime_dim"), PROPRIO_DIM)
    _require_equal(
        "manifest.state_contract.rotation_convention",
        state.get("rotation_convention"),
        ROT6D_CONVENTION,
    )
    _require_equal("manifest.state_contract.source_order", state.get("source_order"), "arm7+eef9+hand10")
    _require_equal("manifest.state_contract.runtime_order", state.get("runtime_order"), "eef9+hand10")
    _require_equal(
        "manifest.state_contract.projection",
        state.get("projection"),
        list(STATE_TO_PROPRIO_INDICES),
    )
    _require_equal(
        "manifest.state_contract.runtime_layout",
        state.get("runtime_layout"),
        list(PROPRIO_LAYOUT),
    )
    _require_equal(
        "manifest.state_contract.runtime_layout_hash",
        state.get("runtime_layout_hash"),
        PROPRIO_LAYOUT_HASH,
    )
    _require_equal(
        "manifest.state_contract.rotation_transposed",
        state.get("rotation_transposed"),
        expected=False,
    )

    action = _require_mapping("manifest.action_contract", manifest.get("action_contract"))
    _require_equal("manifest.action_contract.dimension", action.get("dimension"), ACTION_DIM)
    _require_equal(
        "manifest.action_contract.semantics",
        action.get("semantics"),
        "post_guard_safe_action_in_policy_state_frame",
    )
    _require_equal(
        "manifest.action_contract.rotation_convention",
        action.get("rotation_convention"),
        ROT6D_CONVENTION,
    )

    reference = _require_mapping("manifest.reference_contract", manifest.get("reference_contract"))
    _require_equal(
        "manifest.reference_contract.source_dimension",
        reference.get("source_dimension"),
        REFERENCE_SOURCE_DIM,
    )
    _require_equal(
        "manifest.reference_contract.runtime_dimension",
        reference.get("runtime_dimension"),
        ACTION_DIM,
    )
    _require_equal(
        "manifest.reference_contract.projection",
        reference.get("projection"),
        list(REFERENCE_TO_ACTION_INDICES),
    )
    _require_equal(
        "manifest.reference_contract.source_order",
        reference.get("source_order"),
        "eef9+hand10+arm7",
    )
    _require_equal(
        "manifest.reference_contract.runtime_order",
        reference.get("runtime_order"),
        "eef9+hand10",
    )
    _require_equal(
        "manifest.reference_contract.source_field",
        reference.get("source_field"),
        "vla_reference_action",
    )
    _require_equal(
        "manifest.reference_contract.runtime_field",
        reference.get("runtime_field"),
        "ref_action",
    )
    _require_equal(
        "manifest.reference_contract.learner_consumes_source",
        reference.get("learner_consumes_source"),
        expected=False,
    )
    _require_equal(
        "manifest.partition_boundary_policy",
        manifest.get("partition_boundary_policy"),
        "split_and_drop_nonterminal_tail",
    )
    _require_equal(
        "manifest.terminal_next_state_policy",
        manifest.get("terminal_next_state_policy"),
        "self_state_bootstrap_masked_by_done",
    )
    _require_equal(
        "manifest.recommended_allow_partial",
        manifest.get("recommended_allow_partial"),
        expected=False,
    )


def load_and_validate_bridge_bundle(
    bridge_path: str | os.PathLike[str],
    *,
    expected_fps: float = 10.0,
    expected_feature_contract_fingerprint: str | None = None,
) -> ValidatedBridgeBundle:
    """Load a trusted local bridge pickle and validate every training contract."""

    if expected_feature_contract_fingerprint is not None:
        expected_feature_contract_fingerprint = _require_fingerprint(
            "expected_feature_contract_fingerprint",
            expected_feature_contract_fingerprint,
        )
    path = Path(bridge_path).expanduser().resolve()
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload = _require_mapping("bridge payload", payload)
    _require_equal("payload.schema_name", payload.get("schema_name"), BRIDGE_SCHEMA_NAME)
    _require_equal("payload.schema_version", payload.get("schema_version"), BRIDGE_SCHEMA_VERSION)
    manifest = _require_mapping("payload.manifest", payload.get("manifest"))
    _validate_manifest_contract(manifest, expected_fps=expected_fps)

    fingerprints = {
        name: _require_fingerprint(f"manifest.{name}", manifest.get(name))
        for name in (
            "dataset_fingerprint",
            "feature_contract_fingerprint",
            "bridge_fingerprint",
        )
    }
    if (
        expected_feature_contract_fingerprint is not None
        and fingerprints["feature_contract_fingerprint"] != expected_feature_contract_fingerprint
    ):
        raise ValueError(
            "feature contract fingerprint mismatch: expected "
            f"{expected_feature_contract_fingerprint}, got "
            f"{fingerprints['feature_contract_fingerprint']}"
        )

    segment_count = _require_int("manifest.segment_count", manifest.get("segment_count"), minimum=1)
    replay_step_count = _require_int("manifest.replay_step_count", manifest.get("replay_step_count"), minimum=1)
    source_frame_count = _require_int("manifest.source_frame_count", manifest.get("source_frame_count"), minimum=1)
    if source_frame_count < replay_step_count:
        raise ValueError("manifest.source_frame_count must be >= replay_step_count")
    dropped_count = _require_int(
        "manifest.dropped_boundary_frame_count",
        manifest.get("dropped_boundary_frame_count"),
    )

    episodes_raw = _require_sequence("payload.episodes", payload.get("episodes"))
    if len(episodes_raw) != segment_count:
        raise ValueError(
            f"manifest.segment_count={segment_count} does not match episodes={len(episodes_raw)}"
        )

    episodes: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    total_steps = 0
    actual_dropped_count = 0
    z_dim: int | None = None
    covered_source_frames: set[tuple[int, int]] = set()
    source_episode_ids: set[int] = set()
    terminal_outcomes: dict[int, int] = {}
    for segment_position, episode_value in enumerate(episodes_raw):
        context = f"episodes[{segment_position}]"
        episode = _require_mapping(context, episode_value)
        segment_id = str(episode.get("segment_id", ""))
        if not segment_id:
            raise ValueError(f"{context}.segment_id must be non-empty")
        if segment_id in seen_segments:
            raise ValueError(f"duplicate segment_id {segment_id!r}")
        seen_segments.add(segment_id)
        for name, expected in fingerprints.items():
            _require_equal(f"{context}.{name}", episode.get(name), expected)

        source_episode_index = _require_int(
            f"{context}.source_episode_index",
            episode.get("source_episode_index"),
        )
        source_episode_ids.add(source_episode_index)
        task_episode_id = str(episode.get("task_episode_id", ""))
        if not task_episode_id:
            raise ValueError(f"{context}.task_episode_id must be non-empty")
        partition = str(episode.get("partition", ""))
        if partition not in _PARTITION_SOURCE:
            raise ValueError(f"{context}.partition is unsupported: {partition!r}")
        source_start = _require_int(f"{context}.source_frame_start", episode.get("source_frame_start"))
        source_end = _require_int(
            f"{context}.source_frame_end_inclusive",
            episode.get("source_frame_end_inclusive"),
        )
        if source_end < source_start:
            raise ValueError(f"{context} source frame range is reversed")
        segment_source_frames = {
            (source_episode_index, frame_index)
            for frame_index in range(source_start, source_end + 1)
        }
        overlap = covered_source_frames.intersection(segment_source_frames)
        if overlap:
            raise ValueError(f"{context} source frame range overlaps another segment")
        covered_source_frames.update(segment_source_frames)
        dropped_tail = episode.get("dropped_tail_frame_index")
        if dropped_tail is not None:
            dropped_tail = _require_int(f"{context}.dropped_tail_frame_index", dropped_tail)
            if dropped_tail != source_end:
                raise ValueError(f"{context}.dropped_tail_frame_index must equal source_frame_end_inclusive")
            actual_dropped_count += 1

        steps_raw = _require_sequence(f"{context}.steps", episode.get("steps"))
        if not steps_raw:
            raise ValueError(f"{context}.steps must not be empty")
        expected_source_frame = source_start
        normalized_steps: list[dict[str, Any]] = []
        saw_done = False
        for step_position, step_value in enumerate(steps_raw):
            step_context = f"{context}.steps[{step_position}]"
            step = dict(_require_mapping(step_context, step_value))
            for name, expected in fingerprints.items():
                _require_equal(f"{step_context}.{name}", step.get(name), expected)
            _require_equal(f"{step_context}.segment_id", step.get("segment_id"), segment_id)
            _require_equal(f"{step_context}.source_segment_id", step.get("source_segment_id"), segment_id)
            _require_equal(f"{step_context}.source_partition", step.get("source_partition"), partition)
            _require_equal(f"{step_context}.task_episode_id", step.get("task_episode_id"), task_episode_id)
            _require_equal(
                f"{step_context}.source_episode_index",
                step.get("source_episode_index"),
                source_episode_index,
            )

            episode_id = _require_int(f"{step_context}.episode_id", step.get("episode_id"))
            _require_equal(f"{step_context}.episode_id", episode_id, source_episode_index)
            step_id = _require_int(f"{step_context}.step_id", step.get("step_id"))
            source_frame_index = _require_int(
                f"{step_context}.source_frame_index",
                step.get("source_frame_index"),
            )
            _require_equal(f"{step_context}.step_id", step_id, source_frame_index)
            _require_equal(
                f"{step_context}.source_frame_index",
                source_frame_index,
                expected_source_frame,
            )
            if source_frame_index > source_end:
                raise ValueError(f"{step_context}.source_frame_index exceeds segment frame range")

            done = _require_bool(f"{step_context}.done", step.get("done"))
            if saw_done:
                raise ValueError(f"{step_context} appears after a terminal step")
            if done and step_position != len(steps_raw) - 1:
                raise ValueError(f"{step_context}.done may only be true on the final segment step")
            saw_done = saw_done or done
            next_source_frame_index = _require_int(
                f"{step_context}.next_source_frame_index",
                step.get("next_source_frame_index"),
            )
            expected_next_index = source_frame_index if done else source_frame_index + 1
            _require_equal(
                f"{step_context}.next_source_frame_index",
                next_source_frame_index,
                expected_next_index,
            )

            collection_phase = str(step.get("collection_phase", ""))
            if collection_phase not in {"warmup", "online"}:
                raise ValueError(f"{step_context}.collection_phase is unsupported: {collection_phase!r}")
            source = _require_int(f"{step_context}.source", step.get("source"))
            _require_equal(f"{step_context}.source", source, _PARTITION_SOURCE[partition])
            intervention = _require_bool(
                f"{step_context}.intervention_flag",
                step.get("intervention_flag"),
            )
            _require_equal(
                f"{step_context}.intervention_flag",
                intervention,
                partition == "human_correction",
            )
            success = _require_int(f"{step_context}.success", step.get("success"))
            if success not in (0, 1):
                raise ValueError(f"{step_context}.success must be 0 or 1")

            z_rl = _finite_array(f"{step_context}.z_rl", step.get("z_rl"))
            next_z_rl = _finite_array(f"{step_context}.next_z_rl", step.get("next_z_rl"))
            if z_rl.ndim != 1 or z_rl.size == 0:
                raise ValueError(f"{step_context}.z_rl must be a finite non-empty vector")
            if z_dim is None:
                z_dim = int(z_rl.shape[0])
            if z_rl.shape != (z_dim,) or next_z_rl.shape != (z_dim,):
                raise ValueError(
                    f"{step_context} z_rl/next_z_rl must both have shape ({z_dim},), got "
                    f"{z_rl.shape}/{next_z_rl.shape}"
                )
            step["z_rl"] = z_rl
            step["next_z_rl"] = next_z_rl
            step["proprio"] = _finite_array(
                f"{step_context}.proprio",
                step.get("proprio"),
                (PROPRIO_DIM,),
            )
            step["next_proprio"] = _finite_array(
                f"{step_context}.next_proprio",
                step.get("next_proprio"),
                (PROPRIO_DIM,),
            )
            step["action"] = _finite_array(
                f"{step_context}.action",
                step.get("action"),
                (ACTION_DIM,),
            )
            step["vla_reference_action"] = _finite_array(
                f"{step_context}.vla_reference_action",
                step.get("vla_reference_action"),
                (REFERENCE_SOURCE_DIM,),
            )
            step["ref_action"] = _finite_array(
                f"{step_context}.ref_action",
                step.get("ref_action"),
                (ACTION_DIM,),
            )
            expected_ref_action = step["vla_reference_action"][
                np.asarray(REFERENCE_TO_ACTION_INDICES, dtype=np.int64)
            ]
            if not np.array_equal(step["ref_action"], expected_ref_action):
                raise ValueError(
                    f"{step_context}.ref_action must exactly equal "
                    "vla_reference_action projected with indices [0..18]"
                )
            reward = step.get("reward")
            if isinstance(reward, bool | np.bool_) or not isinstance(reward, int | float | np.number):
                raise TypeError(f"{step_context}.reward must be numeric")
            if not math.isfinite(float(reward)):
                raise ValueError(f"{step_context}.reward must be finite")
            reward = float(reward)
            if not done and (reward != 0.0 or success != 0):
                raise ValueError(f"{step_context} non-terminal reward/success must both be zero")
            if done and reward != float(success):
                raise ValueError(f"{step_context} terminal reward must equal binary success")
            if done:
                if source_episode_index in terminal_outcomes:
                    raise ValueError(f"source episode {source_episode_index} has multiple terminal steps")
                terminal_outcomes[source_episode_index] = success
                if not np.array_equal(step["z_rl"], step["next_z_rl"]):
                    raise ValueError(f"{step_context} terminal z_rl must use the self-state placeholder")
                if not np.array_equal(step["proprio"], step["next_proprio"]):
                    raise ValueError(f"{step_context} terminal proprio must use the self-state placeholder")
            step["reward"] = reward
            step["done"] = done
            step["source"] = source
            step["success"] = success
            step["intervention_flag"] = intervention
            normalized_steps.append(step)
            expected_source_frame += 1

        if dropped_tail is not None and saw_done:
            raise ValueError(f"{context} cannot be terminal and have a dropped boundary tail")
        expected_last_step = source_end - 1 if dropped_tail is not None else source_end
        if normalized_steps[-1]["source_frame_index"] != expected_last_step:
            raise ValueError(
                f"{context} final replay step does not match its source frame range/drop policy"
            )
        for current, following in pairwise(normalized_steps):
            if not np.array_equal(current["next_z_rl"], following["z_rl"]):
                raise ValueError(f"{context} has discontinuous z_rl across adjacent replay steps")
            if not np.array_equal(current["next_proprio"], following["proprio"]):
                raise ValueError(f"{context} has discontinuous proprio across adjacent replay steps")
        episode_copy = dict(episode)
        episode_copy["steps"] = normalized_steps
        episodes.append(episode_copy)
        total_steps += len(normalized_steps)

    if total_steps != replay_step_count:
        raise ValueError(
            f"manifest.replay_step_count={replay_step_count} does not match validated steps={total_steps}"
        )
    if actual_dropped_count > dropped_count:
        raise ValueError(
            "manifest.dropped_boundary_frame_count="
            f"{dropped_count} is smaller than visible dropped segment tails={actual_dropped_count}"
        )
    uncovered_source_frames = source_frame_count - len(covered_source_frames)
    omitted_empty_run_tails = dropped_count - actual_dropped_count
    if uncovered_source_frames != omitted_empty_run_tails:
        raise ValueError(
            "manifest source coverage mismatch: uncovered source frames="
            f"{uncovered_source_frames}, omitted empty-run tails={omitted_empty_run_tails}"
        )
    if set(terminal_outcomes) != source_episode_ids:
        missing = sorted(source_episode_ids - set(terminal_outcomes))
        raise ValueError(f"source episodes without one terminal outcome: {missing}")
    outcome_counts = _require_mapping("manifest.outcome_counts", manifest.get("outcome_counts"))
    expected_outcomes = {
        "success": sum(value == 1 for value in terminal_outcomes.values()),
        "failure": sum(value == 0 for value in terminal_outcomes.values()),
    }
    actual_outcomes = {
        name: _require_int(f"manifest.outcome_counts.{name}", outcome_counts.get(name))
        for name in ("success", "failure")
    }
    if actual_outcomes != expected_outcomes:
        raise ValueError(
            f"manifest.outcome_counts={actual_outcomes} does not match terminal steps={expected_outcomes}"
        )
    assert z_dim is not None
    return ValidatedBridgeBundle(
        manifest=dict(manifest),
        episodes=tuple(episodes),
        z_dim=z_dim,
        dataset_fingerprint=fingerprints["dataset_fingerprint"],
        feature_contract_fingerprint=fingerprints["feature_contract_fingerprint"],
        bridge_fingerprint=fingerprints["bridge_fingerprint"],
    )


def _build_transitions(
    bundle: ValidatedBridgeBundle,
) -> tuple[list[RLTTransition], dict[str, int]]:
    transitions: list[RLTTransition] = []
    segment_transition_counts: dict[str, int] = {}
    for episode in bundle.episodes:
        segment_id = str(episode["segment_id"])
        segment_transitions = build_chunk_transitions_from_episode(
            episode["steps"],
            chunk_len=CHUNK_LEN,
            stride=STRIDE,
            allow_partial=False,
        )
        terminal = build_terminal_aligned_chunk_transition(
            episode["steps"],
            chunk_len=CHUNK_LEN,
        )
        existing_starts = {(item.segment_id, item.step_id) for item in segment_transitions}
        if terminal is not None and (terminal.segment_id, terminal.step_id) not in existing_starts:
            segment_transitions.append(terminal)
        if any(item.segment_id != segment_id for item in segment_transitions):
            raise ValueError(f"chunk builder changed or crossed segment {segment_id!r}")
        segment_transition_counts[segment_id] = len(segment_transitions)
        transitions.extend(segment_transitions)
    if not transitions:
        raise ValueError(
            f"bridge contains no full C={CHUNK_LEN} replay window after segment isolation"
        )
    return transitions, segment_transition_counts


def _validate_rl_config(rl_config: RLTOnlineRLConfig, bundle: ValidatedBridgeBundle) -> None:
    expected = {
        "action_dim": ACTION_DIM,
        "chunk_len": CHUNK_LEN,
        "z_dim": bundle.z_dim,
        "proprio_dim": PROPRIO_DIM,
        "rot6d_convention": ROT6D_CONVENTION,
    }
    mismatches = [
        f"{name}={getattr(rl_config, name)!r}, expected {value!r}"
        for name, value in expected.items()
        if getattr(rl_config, name) != value
    ]
    if mismatches:
        raise ValueError("offline bridge rl_config contract mismatch: " + "; ".join(mismatches))


def _derive_smoke_rl_config(
    bundle: ValidatedBridgeBundle,
    config: OfflineBridgeTrainConfig,
) -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=ACTION_DIM,
        chunk_len=CHUNK_LEN,
        z_dim=bundle.z_dim,
        proprio_dim=PROPRIO_DIM,
        rot6d_convention=ROT6D_CONVENTION,
        delta_action_indices=(),
        actor_hidden_dim=config.actor_hidden_dim,
        actor_num_layers=config.actor_num_layers,
        critic_hidden_dim=config.critic_hidden_dim,
        critic_num_layers=config.critic_num_layers,
        warmup_min_size=1,
    )


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _snapshot_lineage(
    bundle: ValidatedBridgeBundle,
    config: OfflineBridgeTrainConfig,
    *,
    rl_config_was_provided: bool,
) -> dict[str, Any]:
    feature_verified = config.expected_feature_contract_fingerprint is not None
    return {
        "artifact_mode": "production" if config.production_run else "non_production_smoke",
        "production_ready": bool(config.production_run),
        "features_are_real": bool(config.features_are_real),
        "feature_contract": {
            "fingerprint": bundle.feature_contract_fingerprint,
            "expected_fingerprint": config.expected_feature_contract_fingerprint,
            "explicitly_verified": feature_verified,
        },
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "bridge_fingerprint": bundle.bridge_fingerprint,
        "rl_config_was_provided": rl_config_was_provided,
        "bridge_schema_name": BRIDGE_SCHEMA_NAME,
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
    }


def _save_training_artifacts(
    output_dir: Path,
    *,
    state: RLTTrainState,
    rl_config: RLTOnlineRLConfig,
    replay_size: int,
    lineage: Mapping[str, Any],
    production_run: bool,
    system_config: OnlineRLSystemConfig | None,
    final_output_dir: Path,
) -> dict[str, str]:
    global_step = int(jax.device_get(state.global_step))
    actor_version = int(jax.device_get(state.actor_version))
    if production_run:
        if system_config is None:
            raise ValueError("production_run requires the full explicit online RL system config")
        actor_path = output_dir / "actor_snapshot" / "actor_snapshot.pkl"
        checkpoint_path = output_dir / "checkpoints" / "latest.pkl"
    else:
        actor_path = output_dir / "actor_snapshot.pkl"
        checkpoint_path = output_dir / "learner_checkpoint.pkl"
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    actor_rl_config = dataclasses.asdict(relativize_rl_config_paths(rl_config, str(actor_path)))
    checkpoint_rl_config = dataclasses.asdict(relativize_rl_config_paths(rl_config, str(checkpoint_path)))
    actor_payload = {
        "version": actor_version,
        "global_step": global_step,
        "rl_config": actor_rl_config,
        "actor_params": _tree_to_numpy(state.actor_params),
        "bridge_lineage": dict(lineage),
    }
    checkpoint_payload = {
        "rl_config": checkpoint_rl_config,
        "state": {
            "actor_params": _tree_to_numpy(state.actor_params),
            "target_actor_params": _tree_to_numpy(state.target_actor_params),
            "critic_params": _tree_to_numpy(state.critic_params),
            "target_critic_params": _tree_to_numpy(state.target_critic_params),
            "actor_opt_state": _tree_to_numpy(state.actor_opt_state),
            "critic_opt_state": _tree_to_numpy(state.critic_opt_state),
            "rng": _tree_to_numpy(state.rng),
            "global_step": global_step,
            "actor_version": actor_version,
        },
        "progress": {"warmup_ready_adds_total": replay_size},
        "bridge_lineage": dict(lineage),
    }
    _atomic_pickle(actor_path, actor_payload)
    _atomic_pickle(checkpoint_path, checkpoint_payload)
    if production_run:
        _atomic_pickle(
            output_dir / "checkpoints" / f"step_{global_step}.pkl",
            checkpoint_payload,
        )
        final_actor = final_output_dir / "actor_snapshot" / "actor_snapshot.pkl"
        final_checkpoints = final_output_dir / "checkpoints"
        final_replay = final_output_dir / "replay" / "replay_journal.pkl"
        final_wandb = final_output_dir / "wandb"
        resolved_system = dataclasses.replace(
            system_config,
            rl=rl_config,
            actor_service=dataclasses.replace(
                system_config.actor_service,
                snapshot_path=str(final_actor),
            ),
            learner_service=dataclasses.replace(
                system_config.learner_service,
                checkpoint_dir=str(final_checkpoints),
                actor_snapshot_path=str(final_actor),
            ),
            replay=dataclasses.replace(
                system_config.replay,
                journal_path=str(final_replay),
            ),
            monitoring=dataclasses.replace(
                system_config.monitoring,
                wandb_dir=str(final_wandb),
            ),
        )
        config_path = output_dir / "checkpoints" / "online_rl_config.yaml"
        save_system_config_yaml(resolved_system, str(config_path))
        artifacts = {
            "metrics": "metrics.json",
            "actor_snapshot": "actor_snapshot/actor_snapshot.pkl",
            "learner_checkpoint": "checkpoints/latest.pkl",
            "learner_step_checkpoint": f"checkpoints/step_{global_step}.pkl",
            "replay_journal": "replay/replay_journal.pkl",
            "resolved_config": "checkpoints/online_rl_config.yaml",
            "manifest": "manifest.json",
        }
        _atomic_json(
            output_dir / "manifest.json",
            {
                "format_version": 1,
                "artifact_mode": "production",
                "production_ready": True,
                "bridge_lineage": dict(lineage),
                "global_step": global_step,
                "actor_version": actor_version,
                "artifacts": artifacts,
            },
        )
        return artifacts
    return {
        "metrics": "metrics.json",
        "actor_snapshot": "actor_snapshot.pkl",
        "learner_checkpoint": "learner_checkpoint.pkl",
        "replay_journal": "replay_journal.pkl",
    }


def run_offline_bridge_training(
    bridge_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    config: OfflineBridgeTrainConfig | None = None,
    rl_config: RLTOnlineRLConfig | None = None,
    system_config: OnlineRLSystemConfig | None = None,
) -> dict[str, Any]:
    """Build strict C=10 replay and run a bounded number of JAX train steps."""

    config = config or OfflineBridgeTrainConfig()
    bundle = load_and_validate_bridge_bundle(
        bridge_path,
        expected_fps=config.expected_fps,
        expected_feature_contract_fingerprint=config.expected_feature_contract_fingerprint,
    )
    rl_config_was_provided = rl_config is not None
    if config.production_run and not rl_config_was_provided:
        raise ValueError("production_run requires an explicit production rl_config")
    if config.production_run and system_config is None:
        raise ValueError("production_run requires the full explicit online RL system config")
    if rl_config is None:
        rl_config = _derive_smoke_rl_config(bundle, config)
    _validate_rl_config(rl_config, bundle)
    transitions, segment_transition_counts = _build_transitions(bundle)
    if len(transitions) > config.replay_capacity:
        raise ValueError(
            f"replay_capacity={config.replay_capacity} is smaller than transitions={len(transitions)}"
        )

    final_output_dir = Path(output_dir).expanduser().resolve()
    if final_output_dir.exists():
        raise FileExistsError(f"output directory already exists: {final_output_dir}")
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.",
            dir=str(final_output_dir.parent),
        )
    )
    try:
        journal_path = (
            staging_dir / "replay" / "replay_journal.pkl"
            if config.production_run
            else staging_dir / "replay_journal.pkl"
        )
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        replay_manager = ReplayManager(
            config.replay_capacity,
            journal_path=str(journal_path),
            seed=config.seed,
            tensor_contract=ReplayTensorContract(
                z_dim=bundle.z_dim,
                proprio_dim=PROPRIO_DIM,
                chunk_len=CHUNK_LEN,
                action_dim=ACTION_DIM,
            ),
        )
        replay_manager.add_transitions(transitions)

        state, actor, critic = init_train_state(rl_config, rng=jax.random.PRNGKey(config.seed))
        adapter = ActionRepresentationAdapter.from_config(rl_config)
        if adapter is None:
            action_q01 = None
            action_q99 = None
        else:
            action_q01 = jnp.asarray(adapter.stats.q01, dtype=jnp.float32)
            action_q99 = jnp.asarray(adapter.stats.q99, dtype=jnp.float32)
        bc_weight = rl_config.warmup_bc_weight if config.bc_weight is None else config.bc_weight
        q_weight = rl_config.warmup_q_weight if config.q_weight is None else config.q_weight
        delta_weight = rl_config.delta_weight if config.delta_weight is None else config.delta_weight

        history: list[dict[str, float]] = []
        last_metrics: dict[str, float] = {}
        for step_index in range(config.train_steps):
            batch_np = replay_manager.sample_batch(config.batch_size)
            batch_np = prepare_replay_training_batch(batch_np, rl_config)
            if adapter is not None:
                batch_np = adapter.prepare_training_batch(batch_np)
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            state, raw_metrics = train_step(
                state,
                batch,
                actor=actor,
                critic=critic,
                rl_config=rl_config,
                bc_weight=float(bc_weight),
                q_weight=float(q_weight),
                delta_weight=float(delta_weight),
                use_action_adapter=adapter is not None,
                action_q01=action_q01,
                action_q99=action_q99,
            )
            last_metrics = {
                name: float(value)
                for name, value in jax.device_get(raw_metrics).items()
            }
            non_finite = [name for name, value in last_metrics.items() if not math.isfinite(value)]
            if non_finite:
                raise FloatingPointError(f"non-finite train metrics at step {step_index + 1}: {non_finite}")
            if (step_index + 1) % config.metrics_interval == 0 or step_index + 1 == config.train_steps:
                history.append(last_metrics)

        lineage = _snapshot_lineage(
            bundle,
            config,
            rl_config_was_provided=rl_config_was_provided,
        )
        source_step_counts: Counter[str] = Counter()
        for transition in transitions:
            assert transition.valid_mask is not None
            for source in transition.source_chunk[transition.valid_mask]:
                source_step_counts[TransitionSource(int(source)).name.lower()] += 1
        replay_stats = replay_manager.stats()
        artifacts = _save_training_artifacts(
            staging_dir,
            state=state,
            rl_config=rl_config,
            replay_size=len(transitions),
            lineage=lineage,
            production_run=config.production_run,
            system_config=system_config,
            final_output_dir=final_output_dir,
        )
        summary: dict[str, Any] = {
            "status": "ok",
            **lineage,
            "input_bridge": str(Path(bridge_path).expanduser().resolve()),
            "contracts": {
                "state_source_dim": STATE_SOURCE_DIM,
                "proprio_dim": PROPRIO_DIM,
                "action_dim": ACTION_DIM,
                "reference_source_dim": REFERENCE_SOURCE_DIM,
                "z_dim": bundle.z_dim,
                "chunk_len": CHUNK_LEN,
                "stride": STRIDE,
                "fps": config.expected_fps,
                "rot6d_convention": ROT6D_CONVENTION,
            },
            "source": {
                "source_frame_count": int(bundle.manifest["source_frame_count"]),
                "replay_step_count": int(bundle.manifest["replay_step_count"]),
                "segment_count": int(bundle.manifest["segment_count"]),
            },
            "replay": {
                "transition_count": len(transitions),
                "buffer_size": int(replay_stats["size"]),
                "capacity": int(replay_stats["capacity"]),
                "segment_transition_counts": segment_transition_counts,
                "segments_without_full_windows": sorted(
                    segment_id
                    for segment_id, count in segment_transition_counts.items()
                    if count == 0
                ),
                "replay_valid_source_row_counts": dict(sorted(source_step_counts.items())),
            },
            "training": {
                "requested_steps": config.train_steps,
                "completed_steps": int(jax.device_get(state.global_step)),
                "actor_version": int(jax.device_get(state.actor_version)),
                "batch_size": config.batch_size,
                "bc_weight": float(bc_weight),
                "q_weight": float(q_weight),
                "delta_weight": float(delta_weight),
                "last_metrics": last_metrics,
                "history": history,
            },
            "artifacts": artifacts,
        }
        _atomic_json(staging_dir / "metrics.json", summary)
        os.replace(staging_dir, final_output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    returned_summary = dict(summary)
    returned_summary["output_dir"] = str(final_output_dir)
    returned_summary["artifact_paths"] = {
        name: str(final_output_dir / relative_path)
        for name, relative_path in summary["artifacts"].items()
    }
    return returned_summary
