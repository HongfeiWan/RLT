# SPDX-License-Identifier: Apache-2.0

"""Fail-closed LeRobot v3/DAgger to online-RLT replay bridge.

The Teleop exporter intentionally removes ``stop_hold`` rows from the training
view. It also resamples each active partition independently. Consequently, two
adjacent rows with different ``teleop_stack.partition`` values do not prove a
continuous environment transition. This bridge emits every contiguous
partition run as a separate replay episode and drops the non-terminal tail of a
run when no trustworthy next observation exists.

The output of :meth:`ReplayBridgeBundle.to_payload` is deliberately compatible
with ``rlt_online_rl/scripts/build_replay.py``: its top-level ``episodes`` value
contains dictionaries with a ``steps`` list. Extra provenance fields are
preserved in the portable payload and ignored by the current JAX step parser.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import pickle
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from groot_rlt.integration.nero_action_contract import (
    ACTOR_PROPRIO_CHANNEL_NAMES,
    ACTOR_PROPRIO_DIM,
    EXECUTED_ACTION_DIM,
    ROT6D_CONVENTION,
    V3_STATE_DIM,
    V3_TO_ACTOR_PROPRIO_INDICES,
    VLA_REFERENCE_DIM,
    VLA_TO_EXECUTED_ACTION_INDICES,
    bridge_v3_executed_action,
    project_v3_policy_state_to_actor_proprio,
    project_vla_reference_to_executed_action,
    semantic_layout_hash,
)
from groot_rlt.integration.online_stats import modality_from_lerobot_v3_metadata

BRIDGE_SCHEMA_NAME = "groot_rlt.lerobot_v3_dagger_replay"
BRIDGE_SCHEMA_VERSION = 2

_EXPECTED_V3_SCHEMA_VERSION = "teleop_stack.lerobot_v3_dagger.v1"
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED_ROW_KEYS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.state",
    "action",
    "intervention",
    "teleop_stack.is_intervention",
    "teleop_stack.task_episode_id",
    "teleop_stack.partition",
    "teleop_stack.behavior_source",
    "teleop_stack.advantage_label_rule",
    "teleop_stack.terminal_label",
    "next.reward",
    "next.done",
)


class ReplaySource(enum.IntEnum):
    """Numeric values consumed by ``rlt_online_rl.replay.TransitionSource``."""

    BASE = 0
    HUMAN = 2


class TrainingPartition(str, enum.Enum):
    """The only Teleop partitions eligible for RLT training."""

    POLICY_ROLLOUT = "policy_rollout"
    HUMAN_CORRECTION = "human_correction"
    HUMAN_DEMO = "human_demo"


_PARTITION_CONTRACT = {
    TrainingPartition.POLICY_ROLLOUT: {
        "behavior_source": "policy",
        "advantage_label_rule": "pending_value_function",
        "intervention": False,
        "replay_source": ReplaySource.BASE,
    },
    TrainingPartition.HUMAN_CORRECTION: {
        "behavior_source": "human",
        "advantage_label_rule": "forced_positive",
        "intervention": True,
        "replay_source": ReplaySource.HUMAN,
    },
    TrainingPartition.HUMAN_DEMO: {
        "behavior_source": "human",
        "advantage_label_rule": "positive_demo",
        "intervention": False,
        "replay_source": ReplaySource.HUMAN,
    },
}


@dataclasses.dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Stable key passed to an offline Machine-A feature provider."""

    dataset_fingerprint: str
    episode_index: int
    frame_index: int
    task_episode_id: str
    partition: str


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayFrameFeatures:
    """Machine-A outputs needed by the JAX replay schema.

    ``vla_reference_action`` must retain the complete 26D checkpoint output.
    The bridge performs the explicit, validated 26D-to-19D projection.
    """

    z_rl: np.ndarray
    vla_reference_action: np.ndarray


class ReplayFeatureProvider(Protocol):
    """Duck-typed feature provider used without importing GR00T or LeRobot."""

    def __call__(
        self,
        identity: FrameIdentity,
        row: Mapping[str, Any],
    ) -> ReplayFrameFeatures | Mapping[str, Any]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayBridgeStep:
    """One step mapping accepted by ``rlt_online_rl.replay.EpisodeStepRecord``."""

    z_rl: np.ndarray
    proprio: np.ndarray
    vla_reference_action: np.ndarray
    ref_action: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    source: int
    collection_phase: str
    success: int
    intervention_flag: bool
    episode_id: int
    step_id: int
    source_frame_index: int
    next_source_frame_index: int

    def to_mapping(
        self,
        *,
        dataset_fingerprint: str,
        feature_contract_fingerprint: str,
        bridge_fingerprint: str,
        task_episode_id: str,
        partition: str,
        segment_id: str,
    ) -> dict[str, Any]:
        """Return a replay-compatible mapping with immutable source provenance."""

        return {
            "z_rl": self.z_rl.copy(),
            "proprio": self.proprio.copy(),
            "vla_reference_action": self.vla_reference_action.copy(),
            "ref_action": self.ref_action.copy(),
            "action": self.action.copy(),
            "reward": float(self.reward),
            "done": bool(self.done),
            "next_z_rl": self.next_z_rl.copy(),
            "next_proprio": self.next_proprio.copy(),
            "source": int(self.source),
            "collection_phase": self.collection_phase,
            "success": int(self.success),
            "intervention_flag": bool(self.intervention_flag),
            "episode_id": int(self.episode_id),
            "step_id": int(self.step_id),
            "dataset_fingerprint": dataset_fingerprint,
            "feature_contract_fingerprint": feature_contract_fingerprint,
            "bridge_fingerprint": bridge_fingerprint,
            "task_episode_id": task_episode_id,
            "source_episode_index": int(self.episode_id),
            "source_frame_index": int(self.source_frame_index),
            "next_source_frame_index": int(self.next_source_frame_index),
            "source_partition": partition,
            "segment_id": segment_id,
            "source_segment_id": segment_id,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayBridgeSegment:
    """A continuous run whose replay windows must remain isolated."""

    segment_id: str
    episode_index: int
    task_episode_id: str
    partition: str
    source_frame_start: int
    source_frame_end_inclusive: int
    dropped_tail_frame_index: int | None
    steps: tuple[ReplayBridgeStep, ...]

    def to_mapping(
        self,
        *,
        dataset_fingerprint: str,
        feature_contract_fingerprint: str,
        bridge_fingerprint: str,
    ) -> dict[str, Any]:
        """Serialize this segment as one input episode for the JAX builder."""

        return {
            "segment_id": self.segment_id,
            "dataset_fingerprint": dataset_fingerprint,
            "feature_contract_fingerprint": feature_contract_fingerprint,
            "bridge_fingerprint": bridge_fingerprint,
            "source_episode_index": self.episode_index,
            "task_episode_id": self.task_episode_id,
            "partition": self.partition,
            "source_frame_start": self.source_frame_start,
            "source_frame_end_inclusive": self.source_frame_end_inclusive,
            "dropped_tail_frame_index": self.dropped_tail_frame_index,
            "steps": [
                step.to_mapping(
                    dataset_fingerprint=dataset_fingerprint,
                    feature_contract_fingerprint=feature_contract_fingerprint,
                    bridge_fingerprint=bridge_fingerprint,
                    task_episode_id=self.task_episode_id,
                    partition=self.partition,
                    segment_id=self.segment_id,
                )
                for step in self.steps
            ],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayBridgeBundle:
    """Validated portable replay input plus an audit manifest."""

    dataset_fingerprint: str
    feature_contract_fingerprint: str
    bridge_fingerprint: str
    fps: float
    source_frame_count: int
    replay_step_count: int
    dropped_boundary_frame_count: int
    outcome_counts: Mapping[str, int]
    segments: tuple[ReplayBridgeSegment, ...]

    def manifest(self) -> dict[str, Any]:
        """Return JSON-compatible bridge lineage and coverage metadata."""

        return {
            "schema_name": BRIDGE_SCHEMA_NAME,
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "source_format": "lerobot_v3_dagger",
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_contract_fingerprint": self.feature_contract_fingerprint,
            "bridge_fingerprint": self.bridge_fingerprint,
            "fps": float(self.fps),
            "state_contract": {
                "source_dim": V3_STATE_DIM,
                "runtime_dim": ACTOR_PROPRIO_DIM,
                "rotation_convention": ROT6D_CONVENTION,
                "source_order": "arm7+eef9+hand10",
                "runtime_order": "eef9+hand10",
                "projection": list(V3_TO_ACTOR_PROPRIO_INDICES),
                "runtime_layout": list(ACTOR_PROPRIO_CHANNEL_NAMES),
                "runtime_layout_hash": semantic_layout_hash(
                    ACTOR_PROPRIO_CHANNEL_NAMES,
                    rotation_convention=ROT6D_CONVENTION,
                ),
                "rotation_transposed": False,
            },
            "action_contract": {
                "dimension": EXECUTED_ACTION_DIM,
                "semantics": "post_guard_safe_action_in_policy_state_frame",
                "rotation_convention": ROT6D_CONVENTION,
            },
            "reference_contract": {
                "source_dimension": VLA_REFERENCE_DIM,
                "runtime_dimension": EXECUTED_ACTION_DIM,
                "projection": list(VLA_TO_EXECUTED_ACTION_INDICES),
                "source_order": "eef9+hand10+arm7",
                "runtime_order": "eef9+hand10",
                "source_field": "vla_reference_action",
                "runtime_field": "ref_action",
                "learner_consumes_source": False,
            },
            "source_frame_count": int(self.source_frame_count),
            "replay_step_count": int(self.replay_step_count),
            "segment_count": len(self.segments),
            "dropped_boundary_frame_count": int(self.dropped_boundary_frame_count),
            "outcome_counts": dict(self.outcome_counts),
            "partition_boundary_policy": "split_and_drop_nonterminal_tail",
            "terminal_next_state_policy": "self_state_bootstrap_masked_by_done",
            "recommended_allow_partial": False,
        }

    def to_payload(self) -> dict[str, Any]:
        """Return the pickle payload consumed by the current replay script."""

        return {
            "schema_name": BRIDGE_SCHEMA_NAME,
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "manifest": self.manifest(),
            "episodes": [
                segment.to_mapping(
                    dataset_fingerprint=self.dataset_fingerprint,
                    feature_contract_fingerprint=self.feature_contract_fingerprint,
                    bridge_fingerprint=self.bridge_fingerprint,
                )
                for segment in self.segments
            ],
        }


@dataclasses.dataclass(slots=True)
class _ValidatedFrame:
    raw_row: Mapping[str, Any]
    episode_index: int
    frame_index: int
    timestamp: float
    state: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    intervention: bool
    task_episode_id: str
    partition: TrainingPartition
    behavior_source: str
    advantage_label_rule: str
    terminal_label: str
    reward: float
    done: bool
    z_rl: np.ndarray | None = None
    vla_reference_action: np.ndarray | None = None
    ref_action: np.ndarray | None = None


def open_official_lerobot_dataset(
    *args: Any,
    dataset_factory: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Open an official dataset without making LeRobot a required dependency.

    Tests and downstream applications may inject any duck-typed factory. With
    no factory, the import occurs only when this function is called.
    """

    if dataset_factory is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not installed. Install a compatible official LeRobot v3 "
                "package or pass dataset_factory explicitly."
            ) from exc
        dataset_factory = LeRobotDataset
    return dataset_factory(*args, **kwargs)


def _canonical_json_bytes(value: Any, *, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON metadata") from exc


def _update_hash(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _require_fingerprint(name: str, value: str) -> str:
    value = str(value)
    if _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must have form 'sha256:' followed by 64 lowercase hex digits")
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any, *, path: str) -> Any:
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8") if isinstance(value, bytes) else value
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"{path} must contain exactly one scalar, got shape {array.shape}")
    item = array.reshape(-1)[0].item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def _strict_bool(value: Any, *, path: str) -> bool:
    item = _scalar(value, path=path)
    if not isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{path} must be boolean, got {item!r}")
    return bool(item)


def _strict_int(value: Any, *, path: str) -> int:
    item = _scalar(value, path=path)
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
        raise ValueError(f"{path} must be an integer, got {item!r}")
    result = int(item)
    if result < 0:
        raise ValueError(f"{path} must be non-negative, got {result}")
    return result


def _finite_float(value: Any, *, path: str) -> float:
    item = _scalar(value, path=path)
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{path} must be numeric, got {item!r}")
    try:
        result = float(item)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric, got {item!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite, got {result!r}")
    return result


def _strict_string(value: Any, *, path: str, allow_empty: bool = False) -> str:
    item = _scalar(value, path=path)
    if not isinstance(item, str):
        raise ValueError(f"{path} must be a string, got {item!r}")
    if not allow_empty and not item:
        raise ValueError(f"{path} must be non-empty")
    return item


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object, got {type(value).__name__}")
    return value


def _require_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{path} must be an array")
    return value


def _require_int_metadata(value: Any, *, path: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer, got {value!r}")
    if value < (1 if positive else 0):
        comparator = "positive" if positive else "non-negative"
        raise ValueError(f"{path} must be {comparator}, got {value}")
    return int(value)


def _require_extra_feature(
    info_payload: Mapping[str, Any],
    *,
    key: str,
    dtype: str,
) -> None:
    features = _require_mapping(info_payload.get("features"), path="info.features")
    feature = _require_mapping(features.get(key), path=f"info.features.{key}")
    if feature.get("dtype") != dtype:
        raise ValueError(
            f"info.features.{key}.dtype must be {dtype!r}, got {feature.get('dtype')!r}"
        )
    shape = _require_sequence(feature.get("shape"), path=f"info.features.{key}.shape")
    if list(shape) != [1]:
        raise ValueError(f"info.features.{key}.shape must be [1], got {list(shape)!r}")
    if feature.get("names") is not None:
        raise ValueError(f"info.features.{key}.names must be null")


def _validate_metadata(
    info_payload: Mapping[str, Any],
    recap_payload: Mapping[str, Any],
) -> tuple[float, dict[int, Mapping[str, Any]], int]:
    info = _require_mapping(info_payload, path="info")
    recap = _require_mapping(recap_payload, path="teleop_stack_recap")
    modality_from_lerobot_v3_metadata(info, recap)

    if recap.get("schema_version") != _EXPECTED_V3_SCHEMA_VERSION:
        raise ValueError(
            "teleop_stack_recap.schema_version must be "
            f"{_EXPECTED_V3_SCHEMA_VERSION!r}, got {recap.get('schema_version')!r}"
        )
    for key in ("repo_id", "raw_capture_id"):
        if not isinstance(recap.get(key), str) or not recap[key]:
            raise ValueError(f"teleop_stack_recap.{key} must be a non-empty string")
    excluded = _require_sequence(
        recap.get("excluded_partitions"), path="teleop_stack_recap.excluded_partitions"
    )
    if set(excluded) != {"stop_hold", "outside_episode"}:
        raise ValueError(
            "teleop_stack_recap.excluded_partitions must contain exactly "
            "['stop_hold', 'outside_episode']"
        )

    for key, dtype in (
        ("teleop_stack.is_intervention", "bool"),
        ("teleop_stack.task_episode_id", "string"),
        ("teleop_stack.partition", "string"),
        ("teleop_stack.behavior_source", "string"),
        ("teleop_stack.advantage_label_rule", "string"),
        ("teleop_stack.terminal_label", "string"),
        ("next.reward", "float32"),
        ("next.done", "bool"),
    ):
        _require_extra_feature(info, key=key, dtype=dtype)

    total_episodes = _require_int_metadata(
        info.get("total_episodes"), path="info.total_episodes", positive=True
    )
    total_frames = _require_int_metadata(
        info.get("total_frames"), path="info.total_frames", positive=True
    )
    fps = _finite_float(info.get("fps"), path="info.fps")
    recap_fps = _finite_float(recap.get("fps"), path="teleop_stack_recap.fps")
    if fps <= 0.0 or recap_fps <= 0.0 or not math.isclose(fps, recap_fps):
        raise ValueError(
            f"info.fps ({fps}) must equal positive teleop_stack_recap.fps ({recap_fps})"
        )

    sidecar_rows = _require_sequence(recap.get("episodes"), path="teleop_stack_recap.episodes")
    if len(sidecar_rows) != total_episodes:
        raise ValueError(
            f"teleop_stack_recap.episodes has {len(sidecar_rows)} rows, expected {total_episodes}"
        )
    sidecars: dict[int, Mapping[str, Any]] = {}
    for position, value in enumerate(sidecar_rows):
        sidecar = _require_mapping(value, path=f"teleop_stack_recap.episodes[{position}]")
        episode_index = _require_int_metadata(
            sidecar.get("episode_index"),
            path=f"teleop_stack_recap.episodes[{position}].episode_index",
        )
        if episode_index in sidecars:
            raise ValueError(f"duplicate sidecar episode_index {episode_index}")
        label = sidecar.get("terminal_label")
        if label not in {"success", "failure"}:
            raise ValueError(
                f"teleop_stack_recap.episodes[{position}].terminal_label must be success or failure"
            )
        success = sidecar.get("success")
        if not isinstance(success, bool) or success != (label == "success"):
            raise ValueError(
                f"teleop_stack_recap.episodes[{position}].success does not match {label!r}"
            )
        _require_int_metadata(
            sidecar.get("length"),
            path=f"teleop_stack_recap.episodes[{position}].length",
            positive=True,
        )
        for key in ("task_episode_id", "source_episode_dir"):
            if not isinstance(sidecar.get(key), str) or not sidecar[key]:
                raise ValueError(f"teleop_stack_recap.episodes[{position}].{key} must be non-empty")
        sidecars[episode_index] = sidecar
    if set(sidecars) != set(range(total_episodes)):
        raise ValueError("sidecar episode_index values must be contiguous from zero")
    return fps, sidecars, total_frames


def _iter_loader_rows(loader: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(loader, Mapping):
        raise TypeError("loader must yield row mappings, not be a single mapping")
    if hasattr(loader, "__len__") and hasattr(loader, "__getitem__"):
        for index in range(len(loader)):
            row = loader[index]
            if not isinstance(row, Mapping):
                raise TypeError(f"loader[{index}] must be a mapping, got {type(row).__name__}")
            yield row
        return
    try:
        iterator = iter(loader)
    except TypeError as exc:
        raise TypeError("loader must be iterable or implement __len__ and __getitem__") from exc
    for index, row in enumerate(iterator):
        if not isinstance(row, Mapping):
            raise TypeError(f"loader row {index} must be a mapping, got {type(row).__name__}")
        yield row


def _validate_row(row: Mapping[str, Any], *, position: int, fps: float) -> _ValidatedFrame:
    missing = [key for key in _REQUIRED_ROW_KEYS if key not in row]
    if missing:
        raise ValueError(f"loader row {position} is missing required keys: {missing}")
    prefix = f"loader row {position}"
    episode_index = _strict_int(row["episode_index"], path=f"{prefix}.episode_index")
    frame_index = _strict_int(row["frame_index"], path=f"{prefix}.frame_index")
    timestamp = _finite_float(row["timestamp"], path=f"{prefix}.timestamp")
    expected_timestamp = frame_index / fps
    if not math.isclose(timestamp, expected_timestamp, rel_tol=1.0e-5, abs_tol=1.0e-5):
        raise ValueError(
            f"{prefix}.timestamp {timestamp} does not match frame_index/fps {expected_timestamp}"
        )

    partition_value = _strict_string(
        row["teleop_stack.partition"], path=f"{prefix}.teleop_stack.partition"
    )
    try:
        partition = TrainingPartition(partition_value)
    except ValueError as exc:
        raise ValueError(f"{prefix} has non-training partition {partition_value!r}") from exc
    contract = _PARTITION_CONTRACT[partition]

    intervention = _strict_bool(row["intervention"], path=f"{prefix}.intervention")
    legacy_intervention = _strict_bool(
        row["teleop_stack.is_intervention"],
        path=f"{prefix}.teleop_stack.is_intervention",
    )
    expected_intervention = bool(contract["intervention"])
    if intervention != legacy_intervention or intervention != expected_intervention:
        raise ValueError(
            f"{prefix} intervention mismatch: intervention={intervention}, "
            f"teleop_stack.is_intervention={legacy_intervention}, "
            f"partition={partition.value!r} expects {expected_intervention}"
        )

    behavior_source = _strict_string(
        row["teleop_stack.behavior_source"],
        path=f"{prefix}.teleop_stack.behavior_source",
    )
    if behavior_source != contract["behavior_source"]:
        raise ValueError(
            f"{prefix} behavior_source {behavior_source!r} does not match partition "
            f"{partition.value!r}"
        )
    advantage_label_rule = _strict_string(
        row["teleop_stack.advantage_label_rule"],
        path=f"{prefix}.teleop_stack.advantage_label_rule",
    )
    if advantage_label_rule != contract["advantage_label_rule"]:
        raise ValueError(
            f"{prefix} advantage_label_rule {advantage_label_rule!r} does not match "
            f"partition {partition.value!r}"
        )

    terminal_label = _strict_string(
        row["teleop_stack.terminal_label"],
        path=f"{prefix}.teleop_stack.terminal_label",
    )
    if terminal_label not in {"success", "failure"}:
        raise ValueError(f"{prefix}.teleop_stack.terminal_label must be success or failure")
    reward = _finite_float(row["next.reward"], path=f"{prefix}.next.reward")
    done = _strict_bool(row["next.done"], path=f"{prefix}.next.done")

    state = np.asarray(_to_numpy(row["observation.state"]), dtype=np.float32)
    action = np.asarray(_to_numpy(row["action"]), dtype=np.float32)
    proprio = project_v3_policy_state_to_actor_proprio(
        state,
        rotation_convention=ROT6D_CONVENTION,
    )
    executed_action = bridge_v3_executed_action(
        action,
        rotation_convention=ROT6D_CONVENTION,
    )
    return _ValidatedFrame(
        raw_row=row,
        episode_index=episode_index,
        frame_index=frame_index,
        timestamp=timestamp,
        state=state.copy(),
        proprio=proprio,
        action=executed_action,
        intervention=intervention,
        task_episode_id=_strict_string(
            row["teleop_stack.task_episode_id"],
            path=f"{prefix}.teleop_stack.task_episode_id",
        ),
        partition=partition,
        behavior_source=behavior_source,
        advantage_label_rule=advantage_label_rule,
        terminal_label=terminal_label,
        reward=reward,
        done=done,
    )


def _validate_episodes(
    frames: Sequence[_ValidatedFrame],
    *,
    sidecars: Mapping[int, Mapping[str, Any]],
    total_frames: int,
) -> dict[int, list[_ValidatedFrame]]:
    if len(frames) != total_frames:
        raise ValueError(
            f"loader yielded {len(frames)} rows, but info.total_frames is {total_frames}"
        )
    if not frames:
        raise ValueError("loader yielded no rows")
    keys = [(frame.episode_index, frame.frame_index) for frame in frames]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("loader rows must be unique and ordered by (episode_index, frame_index)")

    episodes: dict[int, list[_ValidatedFrame]] = {}
    for frame in frames:
        if frame.episode_index not in sidecars:
            raise ValueError(
                f"loader contains episode_index {frame.episode_index} absent from sidecar"
            )
        episodes.setdefault(frame.episode_index, []).append(frame)
    if set(episodes) != set(sidecars):
        missing = sorted(set(sidecars) - set(episodes))
        raise ValueError(f"loader is missing sidecar episodes {missing}")

    for episode_index, episode_frames in episodes.items():
        sidecar = sidecars[episode_index]
        expected_length = int(sidecar["length"])
        if len(episode_frames) != expected_length:
            raise ValueError(
                f"episode {episode_index} has {len(episode_frames)} rows, "
                f"expected {expected_length}"
            )
        actual_indices = [frame.frame_index for frame in episode_frames]
        if actual_indices != list(range(expected_length)):
            raise ValueError(f"episode {episode_index} frame_index must be contiguous from zero")

        expected_label = str(sidecar["terminal_label"])
        expected_task_episode_id = str(sidecar["task_episode_id"])
        for frame in episode_frames:
            if frame.terminal_label != expected_label:
                raise ValueError(
                    f"episode {episode_index} frame {frame.frame_index} terminal_label does not "
                    "match the sidecar"
                )
            if frame.task_episode_id != expected_task_episode_id:
                raise ValueError(
                    f"episode {episode_index} frame {frame.frame_index} task_episode_id does not "
                    "match the sidecar"
                )
        for frame in episode_frames[:-1]:
            if frame.done or frame.reward != 0.0:
                raise ValueError(
                    f"episode {episode_index} non-terminal frame {frame.frame_index} must have "
                    "next.done=false and next.reward=0"
                )
        terminal = episode_frames[-1]
        expected_reward = 1.0 if expected_label == "success" else 0.0
        if not terminal.done or terminal.reward != expected_reward:
            raise ValueError(
                f"episode {episode_index} terminal frame must have next.done=true and "
                f"next.reward={expected_reward} for {expected_label}"
            )
    return episodes


def _dataset_fingerprint(
    info_payload: Mapping[str, Any],
    recap_payload: Mapping[str, Any],
    frames: Sequence[_ValidatedFrame],
) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, b"groot-rlt-lerobot-v3-semantic-source-v1")
    _update_hash(hasher, _canonical_json_bytes(info_payload, name="info"))
    _update_hash(hasher, _canonical_json_bytes(recap_payload, name="teleop_stack_recap"))
    for frame in frames:
        scalar_payload = {
            "episode_index": frame.episode_index,
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
            "task_episode_id": frame.task_episode_id,
            "partition": frame.partition.value,
            "behavior_source": frame.behavior_source,
            "advantage_label_rule": frame.advantage_label_rule,
            "intervention": frame.intervention,
            "terminal_label": frame.terminal_label,
            "reward": frame.reward,
            "done": frame.done,
        }
        _update_hash(hasher, _canonical_json_bytes(scalar_payload, name="validated row"))
        _update_hash(hasher, np.asarray(frame.state, dtype="<f4").tobytes(order="C"))
        _update_hash(hasher, np.asarray(frame.action, dtype="<f4").tobytes(order="C"))
    return f"sha256:{hasher.hexdigest()}"


def _validated_source_dataset(
    loader: Any,
    *,
    info_payload: Mapping[str, Any],
    recap_payload: Mapping[str, Any],
) -> tuple[
    float,
    dict[int, Mapping[str, Any]],
    list[_ValidatedFrame],
    dict[int, list[_ValidatedFrame]],
    str,
]:
    fps, sidecars, total_frames = _validate_metadata(info_payload, recap_payload)
    frames = [
        _validate_row(row, position=position, fps=fps)
        for position, row in enumerate(_iter_loader_rows(loader))
    ]
    episodes = _validate_episodes(frames, sidecars=sidecars, total_frames=total_frames)
    return (
        fps,
        sidecars,
        frames,
        episodes,
        _dataset_fingerprint(info_payload, recap_payload, frames),
    )


def inspect_lerobot_v3_replay_source(
    loader: Any,
    *,
    info_payload: Mapping[str, Any],
    recap_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate source rows and return the fingerprint needed for first approval.

    This performs the same metadata, row, episode, terminal, action, and state
    validation as :func:`build_lerobot_v3_replay_bundle`, but never invokes a
    feature provider or the 400k policy.
    """

    fps, sidecars, frames, episodes, dataset_fingerprint = _validated_source_dataset(
        loader,
        info_payload=info_payload,
        recap_payload=recap_payload,
    )
    partition_frame_counts = {partition.value: 0 for partition in TrainingPartition}
    partition_run_counts = {partition.value: 0 for partition in TrainingPartition}
    replay_segment_count = 0
    replay_step_count = 0
    dropped_boundary_frame_count = 0
    for episode_frames in episodes.values():
        for run in _partition_runs(episode_frames):
            partition = run[0].partition.value
            partition_run_counts[partition] += 1
            partition_frame_counts[partition] += len(run)
            valid_step_count = len(run) if run[-1].done else len(run) - 1
            if not run[-1].done:
                dropped_boundary_frame_count += 1
            if valid_step_count > 0:
                replay_segment_count += 1
                replay_step_count += valid_step_count
    if replay_segment_count == 0:
        raise ValueError("no replay steps remain after partition-boundary safety filtering")
    outcome_counts = {"success": 0, "failure": 0}
    for sidecar in sidecars.values():
        outcome_counts[str(sidecar["terminal_label"])] += 1
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "fps": fps,
        "source_frame_count": len(frames),
        "episode_count": len(episodes),
        "partition_run_count": sum(partition_run_counts.values()),
        "replay_segment_count": replay_segment_count,
        "replay_step_count": replay_step_count,
        "dropped_boundary_frame_count": dropped_boundary_frame_count,
        "partition_frame_counts": partition_frame_counts,
        "partition_run_counts": partition_run_counts,
        "outcome_counts": outcome_counts,
    }


def _resolve_features(
    result: ReplayFrameFeatures | Mapping[str, Any],
    *,
    identity: FrameIdentity,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(result, ReplayFrameFeatures):
        z_value = result.z_rl
        reference_value = result.vla_reference_action
    elif isinstance(result, Mapping):
        missing = [key for key in ("z_rl", "vla_reference_action") if key not in result]
        if missing:
            raise ValueError(f"feature provider result is missing keys {missing}")
        z_value = result["z_rl"]
        reference_value = result["vla_reference_action"]
    else:
        raise TypeError(
            "feature provider must return ReplayFrameFeatures or a mapping, got "
            f"{type(result).__name__}"
        )

    z_rl = np.asarray(_to_numpy(z_value), dtype=np.float32)
    if z_rl.ndim != 1 or z_rl.size == 0 or not np.isfinite(z_rl).all():
        raise ValueError(
            f"feature provider z_rl for episode={identity.episode_index} "
            f"frame={identity.frame_index} must be a finite non-empty vector, got {z_rl.shape}"
        )
    reference = np.asarray(_to_numpy(reference_value), dtype=np.float32)
    if reference.shape != (VLA_REFERENCE_DIM,):
        raise ValueError(
            f"feature provider vla_reference_action for episode={identity.episode_index} "
            f"frame={identity.frame_index} must have shape ({VLA_REFERENCE_DIM},), "
            f"got {reference.shape}"
        )
    projected = project_vla_reference_to_executed_action(reference)
    return z_rl.copy(), reference.copy(), projected


def _feature_fingerprint(
    dataset_fingerprint: str,
    feature_contract_fingerprint: str,
    frames: Sequence[_ValidatedFrame],
) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, b"groot-rlt-lerobot-v3-replay-features-v2")
    _update_hash(hasher, dataset_fingerprint.encode("ascii"))
    _update_hash(hasher, feature_contract_fingerprint.encode("ascii"))
    for frame in frames:
        assert frame.z_rl is not None
        assert frame.vla_reference_action is not None
        assert frame.ref_action is not None
        _update_hash(
            hasher,
            _canonical_json_bytes(
                {"episode_index": frame.episode_index, "frame_index": frame.frame_index},
                name="feature identity",
            ),
        )
        _update_hash(hasher, np.asarray(frame.z_rl, dtype="<f4").tobytes(order="C"))
        _update_hash(
            hasher,
            np.asarray(frame.vla_reference_action, dtype="<f4").tobytes(order="C"),
        )
        _update_hash(hasher, np.asarray(frame.ref_action, dtype="<f4").tobytes(order="C"))
    return f"sha256:{hasher.hexdigest()}"


def _partition_runs(frames: Sequence[_ValidatedFrame]) -> list[list[_ValidatedFrame]]:
    runs: list[list[_ValidatedFrame]] = []
    for frame in frames:
        if not runs or runs[-1][-1].partition is not frame.partition:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


def _build_segments(
    episodes: Mapping[int, Sequence[_ValidatedFrame]],
    *,
    dataset_fingerprint: str,
    collection_phase: str,
) -> tuple[tuple[ReplayBridgeSegment, ...], int]:
    segments: list[ReplayBridgeSegment] = []
    dropped_boundary_frames = 0
    fingerprint_short = dataset_fingerprint.removeprefix("sha256:")[:16]
    for episode_index, episode_frames in episodes.items():
        for run_index, run in enumerate(_partition_runs(episode_frames)):
            steps: list[ReplayBridgeStep] = []
            dropped_tail: int | None = None
            for offset, frame in enumerate(run):
                if offset + 1 < len(run):
                    next_frame = run[offset + 1]
                elif frame.done:
                    # The critic masks terminal bootstrap. A self-state placeholder
                    # retains the exporter's final real action without inventing a
                    # cross-episode observation.
                    next_frame = frame
                else:
                    dropped_tail = frame.frame_index
                    dropped_boundary_frames += 1
                    continue
                assert frame.z_rl is not None
                assert frame.vla_reference_action is not None
                assert frame.ref_action is not None
                assert next_frame.z_rl is not None
                contract = _PARTITION_CONTRACT[frame.partition]
                steps.append(
                    ReplayBridgeStep(
                        z_rl=frame.z_rl.copy(),
                        proprio=frame.proprio.copy(),
                        vla_reference_action=frame.vla_reference_action.copy(),
                        ref_action=frame.ref_action.copy(),
                        action=frame.action.copy(),
                        reward=frame.reward,
                        done=frame.done,
                        next_z_rl=next_frame.z_rl.copy(),
                        next_proprio=next_frame.proprio.copy(),
                        source=int(contract["replay_source"]),
                        collection_phase=collection_phase,
                        success=int(frame.done and frame.terminal_label == "success"),
                        intervention_flag=frame.intervention,
                        episode_id=episode_index,
                        step_id=frame.frame_index,
                        source_frame_index=frame.frame_index,
                        next_source_frame_index=next_frame.frame_index,
                    )
                )
            if not steps:
                continue
            start = run[0].frame_index
            end = run[-1].frame_index
            segment_id = (
                f"{fingerprint_short}:episode_{episode_index:06d}:run_{run_index:04d}:"
                f"{run[0].partition.value}:frames_{start:06d}_{end:06d}"
            )
            segments.append(
                ReplayBridgeSegment(
                    segment_id=segment_id,
                    episode_index=episode_index,
                    task_episode_id=run[0].task_episode_id,
                    partition=run[0].partition.value,
                    source_frame_start=start,
                    source_frame_end_inclusive=end,
                    dropped_tail_frame_index=dropped_tail,
                    steps=tuple(steps),
                )
            )
    if not segments:
        raise ValueError("no replay steps remain after partition-boundary safety filtering")
    return tuple(segments), dropped_boundary_frames


def build_lerobot_v3_replay_bundle(
    loader: Any,
    *,
    info_payload: Mapping[str, Any],
    recap_payload: Mapping[str, Any],
    feature_provider: ReplayFeatureProvider,
    feature_contract_fingerprint: str,
    expected_dataset_fingerprint: str | None = None,
    collection_phase: str = "warmup",
) -> ReplayBridgeBundle:
    """Validate an official v3/DAgger dataset and build portable replay steps.

    Args:
        loader: Official ``LeRobotDataset`` or any duck-typed row loader.
        info_payload: Parsed ``meta/info.json``.
        recap_payload: Parsed ``meta/teleop_stack_recap.json``.
        feature_provider: Produces a finite RL token and complete 26D VLA
            reference for each validated source row.
        feature_contract_fingerprint: SHA-256 identity of the 400k checkpoint,
            representation encoder, processor, and feature-serving contract.
        expected_dataset_fingerprint: Optional previously approved semantic
            source fingerprint. A mismatch aborts before feature generation.
        collection_phase: Replay phase, either ``warmup`` or ``online``.

    Returns:
        A bridge bundle whose payload is accepted by the current replay script.

    Raises:
        ValueError: If metadata, row semantics, terminal labels, tensor shapes,
            fingerprints, or feature outputs violate the production contract.
    """

    if collection_phase not in {"warmup", "online"}:
        raise ValueError("collection_phase must be 'warmup' or 'online'")
    feature_contract_fingerprint = _require_fingerprint(
        "feature_contract_fingerprint", feature_contract_fingerprint
    )
    if expected_dataset_fingerprint is not None:
        expected_dataset_fingerprint = _require_fingerprint(
            "expected_dataset_fingerprint", expected_dataset_fingerprint
        )

    fps, sidecars, frames, episodes, dataset_fingerprint = _validated_source_dataset(
        loader,
        info_payload=info_payload,
        recap_payload=recap_payload,
    )
    if (
        expected_dataset_fingerprint is not None
        and dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise ValueError(
            "dataset fingerprint mismatch: "
            f"expected {expected_dataset_fingerprint}, got {dataset_fingerprint}"
        )

    z_dim: int | None = None
    for frame in frames:
        identity = FrameIdentity(
            dataset_fingerprint=dataset_fingerprint,
            episode_index=frame.episode_index,
            frame_index=frame.frame_index,
            task_episode_id=frame.task_episode_id,
            partition=frame.partition.value,
        )
        try:
            result = feature_provider(identity, frame.raw_row)
            (
                frame.z_rl,
                frame.vla_reference_action,
                frame.ref_action,
            ) = _resolve_features(result, identity=identity)
        except Exception as exc:
            raise ValueError(
                f"feature generation failed for episode={frame.episode_index} "
                f"frame={frame.frame_index}: {exc}"
            ) from exc
        if z_dim is None:
            z_dim = int(frame.z_rl.shape[0])
        elif frame.z_rl.shape != (z_dim,):
            raise ValueError(
                f"feature provider changed z_rl shape at episode={frame.episode_index} "
                f"frame={frame.frame_index}: got {frame.z_rl.shape}, expected ({z_dim},)"
            )

    bridge_fingerprint = _feature_fingerprint(
        dataset_fingerprint,
        feature_contract_fingerprint,
        frames,
    )
    segments, dropped_boundary_frames = _build_segments(
        episodes,
        dataset_fingerprint=dataset_fingerprint,
        collection_phase=collection_phase,
    )
    outcome_counts = {"success": 0, "failure": 0}
    for sidecar in sidecars.values():
        outcome_counts[str(sidecar["terminal_label"])] += 1
    replay_step_count = sum(len(segment.steps) for segment in segments)
    return ReplayBridgeBundle(
        dataset_fingerprint=dataset_fingerprint,
        feature_contract_fingerprint=feature_contract_fingerprint,
        bridge_fingerprint=bridge_fingerprint,
        fps=fps,
        source_frame_count=len(frames),
        replay_step_count=replay_step_count,
        dropped_boundary_frame_count=dropped_boundary_frames,
        outcome_counts=outcome_counts,
        segments=segments,
    )


def write_replay_bundle(
    bundle: ReplayBridgeBundle,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a portable replay payload for ``build_replay.py``."""

    path = Path(output_path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"replay bridge output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as stream:
            pickle.dump(bundle.to_payload(), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path
