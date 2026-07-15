from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import enum
import http
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from itertools import pairwise
import json
import logging
import os
import pickle
import threading
import time
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from openpi_client import msgpack_numpy

from rlt_online_rl.runtime_logging import append_jsonl

logger = logging.getLogger(__name__)


ArrayDict = dict[str, np.ndarray]


class TransitionSource(enum.IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


DEFAULT_COLLECTION_PHASE = "unknown"
COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


def collection_phase_to_id(phase: str) -> int:
    phase_name = str(phase).split(":", 1)[0].lower()
    if phase_name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if phase_name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def _ensure_array(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


_VALID_TRANSITION_SOURCES = frozenset(int(source) for source in TransitionSource)


def _prefix_mask(valid_steps: int, chunk_len: int) -> np.ndarray:
    if valid_steps < 0 or valid_steps > chunk_len:
        raise ValueError(f"valid_steps must be in [0, {chunk_len}], got {valid_steps}")
    mask = np.zeros((chunk_len,), dtype=np.bool_)
    mask[:valid_steps] = True
    return mask


def _coerce_prefix_mask(name: str, value: Any, *, chunk_len: int) -> np.ndarray:
    raw = _ensure_array(value)
    if raw.dtype != np.dtype(np.bool_) and not np.isin(raw, (0, 1)).all():
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    mask = raw.astype(np.bool_, copy=False)
    if mask.shape != (chunk_len,):
        raise ValueError(f"{name} has shape {mask.shape}, expected ({chunk_len},)")
    false_positions = np.flatnonzero(~mask)
    if false_positions.size and bool(mask[int(false_positions[0]) :].any()):
        raise ValueError(f"{name} must be a contiguous true prefix followed by false padding")
    return mask


def _infer_legacy_valid_mask(data: dict[str, Any], *, chunk_len: int) -> np.ndarray:
    """Accept clearly full legacy records and reject ambiguous padded tails.

    Legacy journals did not persist a validity mask. A zero-valued suffix may be
    padding or a legitimate hold action. Silently choosing either interpretation
    changes the actor target, so ambiguous records fail closed and must be
    regenerated.
    """

    action = _ensure_array(data["action_chunk"], dtype=np.float32)
    reference = _ensure_array(_mapping_reference(data, "ref_chunk", "reference_action_chunk"), dtype=np.float32)
    rewards = _ensure_array(data["rewards"], dtype=np.float32)
    row_has_evidence = (
        np.any(action != 0.0, axis=-1)
        | np.any(reference != 0.0, axis=-1)
        | (rewards != 0.0)
    )
    if "source_chunk" in data:
        source_chunk = _ensure_array(data["source_chunk"])
        row_has_evidence |= source_chunk != int(TransitionSource.BASE)
    if not bool(row_has_evidence[-1]):
        raise ValueError(
            "legacy replay record has an ambiguous zero-valued tail and no valid_mask; "
            "regenerate the journal with explicit masks"
        )
    return np.ones((chunk_len,), dtype=np.bool_)


def _infer_legacy_next_reference_valid_mask(
    data: dict[str, Any],
    next_ref_chunk: np.ndarray,
    *,
    chunk_len: int,
) -> np.ndarray:
    if bool(data["done"]):
        return np.zeros((chunk_len,), dtype=np.bool_)
    if bool(np.any(next_ref_chunk[-1] != 0.0)):
        return np.ones((chunk_len,), dtype=np.bool_)
    raise ValueError(
        "legacy non-terminal replay record has an ambiguous next-reference tail and no "
        "next_reference_valid_mask; regenerate the journal with explicit masks"
    )


def _mapping_reference(data: dict[str, Any], key: str, alias: str) -> Any:
    if key in data:
        return data[key]
    if alias in data:
        return data[alias]
    raise KeyError(f"replay record is missing both {key!r} and {alias!r}")


@dataclasses.dataclass(slots=True)
class EpisodeStepRecord:
    """A single environment step used to build chunk transitions."""

    z_rl: np.ndarray
    proprio: np.ndarray
    ref_action: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    source: int = int(TransitionSource.RL)
    collection_phase: str = DEFAULT_COLLECTION_PHASE
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0
    segment_id: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EpisodeStepRecord:
        return cls(
            z_rl=_ensure_array(data["z_rl"], dtype=np.float32),
            proprio=_ensure_array(data["proprio"], dtype=np.float32),
            ref_action=_ensure_array(data["ref_action"], dtype=np.float32),
            action=_ensure_array(data["action"], dtype=np.float32),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            next_z_rl=_ensure_array(data["next_z_rl"], dtype=np.float32),
            next_proprio=_ensure_array(data["next_proprio"], dtype=np.float32),
            source=int(data.get("source", int(TransitionSource.RL))),
            collection_phase=str(data.get("collection_phase", DEFAULT_COLLECTION_PHASE)),
            success=int(data.get("success", 0)),
            intervention_flag=bool(data.get("intervention_flag", False)),
            episode_id=int(data.get("episode_id", 0)),
            step_id=int(data.get("step_id", 0)),
            segment_id=None if data.get("segment_id") is None else str(data["segment_id"]),
        )


@dataclasses.dataclass(slots=True)
class RLTTransition:
    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    action_chunk: np.ndarray
    rewards: np.ndarray
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source: int
    source_chunk: np.ndarray
    collection_phase: str
    success: int
    intervention_flag: bool
    episode_id: int
    step_id: int
    valid_mask: np.ndarray | None = None
    reference_valid_mask: np.ndarray | None = None
    next_reference_valid_mask: np.ndarray | None = None
    segment_id: str | None = None

    def __post_init__(self) -> None:
        self.z_rl = _ensure_array(self.z_rl, dtype=np.float32)
        self.proprio = _ensure_array(self.proprio, dtype=np.float32)
        self.ref_chunk = _ensure_array(self.ref_chunk, dtype=np.float32)
        self.action_chunk = _ensure_array(self.action_chunk, dtype=np.float32)
        self.rewards = _ensure_array(self.rewards, dtype=np.float32)
        self.next_z_rl = _ensure_array(self.next_z_rl, dtype=np.float32)
        self.next_proprio = _ensure_array(self.next_proprio, dtype=np.float32)
        self.next_ref_chunk = _ensure_array(self.next_ref_chunk, dtype=np.float32)

        if self.ref_chunk.ndim != 2:
            raise ValueError(f"ref_chunk must be rank 2, got shape {self.ref_chunk.shape}")
        chunk_len, action_dim = self.ref_chunk.shape
        action_shape = (chunk_len, action_dim)
        if self.action_chunk.shape != action_shape:
            raise ValueError(f"action_chunk has shape {self.action_chunk.shape}, expected {action_shape}")
        if self.next_ref_chunk.shape != action_shape:
            raise ValueError(f"next_ref_chunk has shape {self.next_ref_chunk.shape}, expected {action_shape}")
        if self.rewards.shape != (chunk_len,):
            raise ValueError(f"rewards has shape {self.rewards.shape}, expected ({chunk_len},)")

        self.source = int(self.source)
        if self.source not in _VALID_TRANSITION_SOURCES:
            raise ValueError(f"source must be one of {sorted(_VALID_TRANSITION_SOURCES)}, got {self.source}")
        raw_source_chunk = _ensure_array(self.source_chunk)
        if raw_source_chunk.shape != (chunk_len,):
            raise ValueError(f"source_chunk has shape {raw_source_chunk.shape}, expected ({chunk_len},)")
        if not np.isin(raw_source_chunk, tuple(_VALID_TRANSITION_SOURCES)).all():
            raise ValueError(f"source_chunk contains values outside {sorted(_VALID_TRANSITION_SOURCES)}")
        self.source_chunk = raw_source_chunk.astype(np.uint8, copy=False)

        self.valid_mask = _coerce_prefix_mask(
            "valid_mask",
            np.ones((chunk_len,), dtype=np.bool_) if self.valid_mask is None else self.valid_mask,
            chunk_len=chunk_len,
        )
        self.reference_valid_mask = _coerce_prefix_mask(
            "reference_valid_mask",
            self.valid_mask if self.reference_valid_mask is None else self.reference_valid_mask,
            chunk_len=chunk_len,
        )
        self.next_reference_valid_mask = _coerce_prefix_mask(
            "next_reference_valid_mask",
            (
                np.zeros((chunk_len,), dtype=np.bool_)
                if self.next_reference_valid_mask is None and bool(self.done)
                else np.ones((chunk_len,), dtype=np.bool_)
                if self.next_reference_valid_mask is None
                else self.next_reference_valid_mask
            ),
            chunk_len=chunk_len,
        )
        if not bool(self.valid_mask.any()):
            raise ValueError("valid_mask must contain at least one valid action step")
        if not np.array_equal(self.reference_valid_mask, self.valid_mask):
            raise ValueError("reference_valid_mask must equal valid_mask for aligned executed/reference chunks")
        if bool(self.done) and bool(self.next_reference_valid_mask.any()):
            raise ValueError("terminal transitions must have an all-false next_reference_valid_mask")

        for name in (
            "z_rl",
            "proprio",
            "ref_chunk",
            "action_chunk",
            "rewards",
            "next_z_rl",
            "next_proprio",
            "next_ref_chunk",
        ):
            value = np.asarray(getattr(self, name))
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

        padding = ~self.valid_mask
        if bool(np.any(self.action_chunk[padding] != 0.0)):
            raise ValueError("action_chunk must be zero in padded rows")
        if bool(np.any(self.ref_chunk[padding] != 0.0)):
            raise ValueError("ref_chunk must be zero in padded rows")
        if bool(np.any(self.rewards[padding] != 0.0)):
            raise ValueError("rewards must be zero in padded rows")
        # Masked next-reference values remain available for audit/parity. The
        # learner canonicalizes them to zero before either target network sees
        # them, so terminal collector payloads stay backward compatible.

        self.done = bool(self.done)
        self.collection_phase = str(self.collection_phase)
        self.success = int(self.success)
        self.intervention_flag = bool(self.intervention_flag)
        self.episode_id = int(self.episode_id)
        self.step_id = int(self.step_id)
        self.segment_id = None if self.segment_id is None else str(self.segment_id)

    @property
    def reference_action_chunk(self) -> np.ndarray:
        """Alias matching the LeRobot v3 replay contract name."""

        return self.ref_chunk

    @property
    def next_reference_action_chunk(self) -> np.ndarray:
        """Alias matching the LeRobot v3 replay contract name."""

        return self.next_ref_chunk

    def to_numpy(self) -> ArrayDict:
        return {
            "z_rl": _ensure_array(self.z_rl, dtype=np.float16),
            "proprio": _ensure_array(self.proprio, dtype=np.float32),
            "ref_chunk": _ensure_array(self.ref_chunk, dtype=np.float32),
            "action_chunk": _ensure_array(self.action_chunk, dtype=np.float32),
            "rewards": _ensure_array(self.rewards, dtype=np.float32),
            "done": _ensure_array(self.done, dtype=np.bool_),
            "next_z_rl": _ensure_array(self.next_z_rl, dtype=np.float16),
            "next_proprio": _ensure_array(self.next_proprio, dtype=np.float32),
            "next_ref_chunk": _ensure_array(self.next_ref_chunk, dtype=np.float32),
            "source": _ensure_array(self.source, dtype=np.uint8),
            "source_chunk": _ensure_array(self.source_chunk, dtype=np.uint8),
            "valid_mask": _ensure_array(self.valid_mask, dtype=np.bool_),
            "reference_valid_mask": _ensure_array(self.reference_valid_mask, dtype=np.bool_),
            "next_reference_valid_mask": _ensure_array(self.next_reference_valid_mask, dtype=np.bool_),
            "collection_phase_id": _ensure_array(collection_phase_to_id(self.collection_phase), dtype=np.uint8),
            "success": _ensure_array(self.success, dtype=np.int8),
            "intervention_flag": _ensure_array(self.intervention_flag, dtype=np.bool_),
            "episode_id": _ensure_array(self.episode_id, dtype=np.int32),
            "step_id": _ensure_array(self.step_id, dtype=np.int32),
        }

    def to_journal_record(self) -> dict[str, Any]:
        return {
            **self.to_numpy(),
            "collection_phase": self.collection_phase,
            "segment_id": self.segment_id,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RLTTransition:
        ref_chunk = _ensure_array(_mapping_reference(data, "ref_chunk", "reference_action_chunk"), dtype=np.float32)
        next_ref_chunk = _ensure_array(
            _mapping_reference(data, "next_ref_chunk", "next_reference_action_chunk"),
            dtype=np.float32,
        )
        if ref_chunk.ndim != 2:
            raise ValueError(f"ref_chunk must be rank 2, got shape {ref_chunk.shape}")
        chunk_len = ref_chunk.shape[0]
        valid_mask = (
            _infer_legacy_valid_mask(data, chunk_len=chunk_len)
            if "valid_mask" not in data
            else data["valid_mask"]
        )
        return cls(
            z_rl=_ensure_array(data["z_rl"], dtype=np.float32),
            proprio=_ensure_array(data["proprio"], dtype=np.float32),
            ref_chunk=ref_chunk,
            action_chunk=_ensure_array(data["action_chunk"], dtype=np.float32),
            rewards=_ensure_array(data["rewards"], dtype=np.float32),
            done=bool(data["done"]),
            next_z_rl=_ensure_array(data["next_z_rl"], dtype=np.float32),
            next_proprio=_ensure_array(data["next_proprio"], dtype=np.float32),
            next_ref_chunk=next_ref_chunk,
            source=int(data["source"]),
            source_chunk=_ensure_array(
                data.get(
                    "source_chunk",
                    np.full((chunk_len,), int(data["source"]), dtype=np.uint8),
                ),
                dtype=np.uint8,
            ),
            collection_phase=str(data.get("collection_phase", DEFAULT_COLLECTION_PHASE)),
            success=int(data.get("success", 0)),
            intervention_flag=bool(data.get("intervention_flag", False)),
            episode_id=int(data.get("episode_id", 0)),
            step_id=int(data.get("step_id", 0)),
            valid_mask=valid_mask,
            reference_valid_mask=data.get("reference_valid_mask", valid_mask),
            next_reference_valid_mask=(
                data["next_reference_valid_mask"]
                if "next_reference_valid_mask" in data
                else _infer_legacy_next_reference_valid_mask(
                    data,
                    next_ref_chunk,
                    chunk_len=chunk_len,
                )
            ),
            segment_id=None if data.get("segment_id") is None else str(data["segment_id"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayTensorContract:
    """Exact tensor shapes accepted by an online-RL replay journal."""

    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int

    def __post_init__(self) -> None:
        for name in ("z_dim", "proprio_dim", "chunk_len", "action_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def validate(self, transition: RLTTransition, *, context: str) -> None:
        expected_shapes = {
            "z_rl": (self.z_dim,),
            "proprio": (self.proprio_dim,),
            "ref_chunk": (self.chunk_len, self.action_dim),
            "action_chunk": (self.chunk_len, self.action_dim),
            "rewards": (self.chunk_len,),
            "next_z_rl": (self.z_dim,),
            "next_proprio": (self.proprio_dim,),
            "next_ref_chunk": (self.chunk_len, self.action_dim),
            "source_chunk": (self.chunk_len,),
            "valid_mask": (self.chunk_len,),
            "reference_valid_mask": (self.chunk_len,),
            "next_reference_valid_mask": (self.chunk_len,),
        }
        for name, expected_shape in expected_shapes.items():
            value = np.asarray(getattr(transition, name))
            if value.shape != expected_shape:
                raise ValueError(f"{context} {name} has shape {value.shape}, expected {expected_shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{context} {name} contains non-finite values")
        for name in ("ref_chunk", "action_chunk", "next_ref_chunk"):
            value = np.asarray(getattr(transition, name))
            if value.dtype != np.dtype(np.float32):
                raise ValueError(f"{context} {name} has dtype {value.dtype}, expected float32")


@dataclasses.dataclass(slots=True)
class RawEpisodeStep:
    observation_idx: int
    next_observation_idx: int
    action: np.ndarray
    ref_action: np.ndarray
    reward: float
    done: bool
    source: int = int(TransitionSource.RL)
    collection_phase: str = DEFAULT_COLLECTION_PHASE
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0
    actor_param_version: int = -1


@dataclasses.dataclass(slots=True)
class RawEpisodeChunk:
    episode_id: int
    chunk_step_id: int
    observation_idx: int
    step_start: int
    step_stop: int
    source: int
    collection_phase: str = DEFAULT_COLLECTION_PHASE
    done: bool = False
    success: int = 0
    drop_transition: bool = False
    start_z_rl: np.ndarray | None = None
    start_proprio: np.ndarray | None = None
    start_ref_chunk: np.ndarray | None = None


@dataclasses.dataclass(slots=True)
class RawEpisodeTrace:
    episode_id: int
    chunk_len: int
    observations: list[dict[str, Any]]
    steps: list[RawEpisodeStep]
    chunks: list[RawEpisodeChunk]
    policy_start_steps: list[int] = dataclasses.field(default_factory=list)
    summary: dict[str, Any] = dataclasses.field(default_factory=dict)


def raw_episode_dir_from_journal(journal_path: str) -> str:
    return os.path.join(os.path.dirname(journal_path) or ".", "episodes")


def raw_episode_path_for(journal_path: str, episode_id: int, *, suffix: str) -> str:
    filename = f"episode_{int(episode_id):06d}_{suffix}.pkl"
    return os.path.join(raw_episode_dir_from_journal(journal_path), filename)


def save_raw_episode(trace: RawEpisodeTrace, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    return path


def _pad_stack(
    values: list[np.ndarray],
    start: int,
    length: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    padded = np.zeros((length, *shape), dtype=dtype)
    for i in range(length):
        idx = start + i
        if idx >= len(values):
            break
        padded[i] = values[idx]
    return padded


def _pad_rewards(rewards: list[float], start: int, length: int) -> np.ndarray:
    padded = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        idx = start + i
        if idx >= len(rewards):
            break
        padded[i] = float(rewards[idx])
    return padded


def _resolve_chunk_source(steps: list[EpisodeStepRecord]) -> tuple[int, bool]:
    intervention = any(step.intervention_flag for step in steps)
    source_values = {int(step.source) for step in steps}
    has_human = int(TransitionSource.HUMAN) in source_values
    has_policy = any(
        source in source_values
        for source in (
            int(TransitionSource.BASE),
            int(TransitionSource.RL),
            int(TransitionSource.MIXED),
        )
    )
    if int(TransitionSource.MIXED) in source_values or (has_human and has_policy):
        return int(TransitionSource.MIXED), intervention
    if has_human or intervention:
        return int(TransitionSource.HUMAN), intervention
    return int(steps[0].source), intervention


def _build_chunk_transition(
    steps: list[EpisodeStepRecord],
    *,
    start: int,
    chunk_len: int,
) -> RLTTransition:
    current = steps[start]
    end = min(start + chunk_len, len(steps))
    window = steps[start:end]
    last = window[-1]
    ref_actions = [step.ref_action for step in steps]
    actions = [step.action for step in steps]
    rewards = [step.reward for step in steps]
    action_shape = steps[0].action.shape
    ref_shape = steps[0].ref_action.shape

    ref_chunk = _pad_stack(ref_actions, start, chunk_len, ref_shape, np.float32)
    action_chunk = _pad_stack(actions, start, chunk_len, action_shape, np.float32)
    reward_chunk = _pad_rewards(rewards, start, chunk_len)
    next_ref_chunk = _pad_stack(ref_actions, start + chunk_len, chunk_len, ref_shape, np.float32)
    source, intervention = _resolve_chunk_source(window)
    source_chunk = _pad_stack(
        [np.asarray(step.source, dtype=np.uint8) for step in steps],
        start,
        chunk_len,
        (),
        np.uint8,
    )
    done = bool(any(step.done for step in window) or last.done)
    valid_mask = _prefix_mask(len(window), chunk_len)
    next_valid_steps = 0 if done else min(chunk_len, max(len(steps) - (start + chunk_len), 0))
    next_reference_valid_mask = _prefix_mask(next_valid_steps, chunk_len)
    return RLTTransition(
        z_rl=current.z_rl,
        proprio=current.proprio,
        ref_chunk=ref_chunk,
        action_chunk=action_chunk,
        rewards=reward_chunk,
        done=done,
        next_z_rl=last.next_z_rl,
        next_proprio=last.next_proprio,
        next_ref_chunk=next_ref_chunk,
        source=source,
        source_chunk=source_chunk,
        collection_phase=current.collection_phase,
        success=int(last.success),
        intervention_flag=intervention,
        episode_id=current.episode_id,
        step_id=current.step_id,
        valid_mask=valid_mask,
        reference_valid_mask=valid_mask,
        next_reference_valid_mask=next_reference_valid_mask,
        segment_id=current.segment_id,
    )


def _split_contiguous_segments(steps: list[EpisodeStepRecord]) -> list[list[EpisodeStepRecord]]:
    if not steps:
        return []

    # Older step logs omitted step_id and decode as an all-zero sequence. In that
    # one case, continuity must be established by the remaining metadata.
    step_ids_are_informative = any(step.step_id != 0 for step in steps)
    segments: list[list[EpisodeStepRecord]] = [[steps[0]]]
    for previous, current in pairwise(steps):
        explicit_segment_break = (
            previous.segment_id is not None
            or current.segment_id is not None
        ) and previous.segment_id != current.segment_id
        discontinuous = (
            bool(previous.done)
            or current.episode_id != previous.episode_id
            or current.collection_phase != previous.collection_phase
            or explicit_segment_break
            or (step_ids_are_informative and current.step_id != previous.step_id + 1)
        )
        if discontinuous:
            segments.append([])
        segments[-1].append(current)
    return segments


def build_chunk_transitions_from_episode(
    episode_steps: Iterable[EpisodeStepRecord | dict[str, Any]],
    *,
    chunk_len: int,
    stride: int = 2,
    allow_partial: bool = True,
) -> list[RLTTransition]:
    """Build chunk transitions from step-level episode logs.

    Assumption:
    - each input step corresponds to one environment step
    - `next_z_rl` / `next_proprio` refer to the state after that step
    - `ref_action` is the step-level executed-dimension reference action
    """

    steps = [
        step if isinstance(step, EpisodeStepRecord) else EpisodeStepRecord.from_mapping(step) for step in episode_steps
    ]
    if not steps:
        return []

    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    transitions: list[RLTTransition] = []
    for segment in _split_contiguous_segments(steps):
        for start in range(0, len(segment), stride):
            is_partial = start + chunk_len > len(segment)
            if start + chunk_len >= len(segment) and not segment[-1].done:
                # A truncated non-terminal segment has no safe bootstrap
                # reference. It must not borrow one across a gap or boundary.
                break
            if is_partial and (not allow_partial or not segment[-1].done):
                break
            transitions.append(_build_chunk_transition(segment, start=start, chunk_len=chunk_len))

    return transitions


def build_terminal_aligned_chunk_transition(
    episode_steps: Iterable[EpisodeStepRecord | dict[str, Any]],
    *,
    chunk_len: int,
) -> RLTTransition | None:
    steps = [
        step if isinstance(step, EpisodeStepRecord) else EpisodeStepRecord.from_mapping(step) for step in episode_steps
    ]
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    segments = _split_contiguous_segments(steps)
    if not segments:
        return None
    terminal_segment = segments[-1]
    if len(terminal_segment) < chunk_len or not terminal_segment[-1].done:
        return None
    start = len(terminal_segment) - chunk_len
    return _build_chunk_transition(terminal_segment, start=start, chunk_len=chunk_len)


class ReplayBuffer:
    """CPU ring buffer for chunk transitions."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 0,
        sample_strategy: str = "uniform",
        recent_episode_window: int = 20,
        recent_online_ratio: float = 0.4,
        warmup_demo_ratio: float = 0.3,
        human_intervention_ratio: float = 0.2,
    ):
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._sample_strategy = sample_strategy
        self._recent_episode_window = int(recent_episode_window)
        self._recent_online_ratio = float(recent_online_ratio)
        self._warmup_demo_ratio = float(warmup_demo_ratio)
        self._human_intervention_ratio = float(human_intervention_ratio)
        self._storage: dict[str, np.ndarray] | None = None
        self._position = 0
        self._size = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    def _initialize_storage(self, record: ArrayDict) -> None:
        self._storage = {}
        for key, value in record.items():
            value = _ensure_array(value)
            self._storage[key] = np.empty((self.capacity, *value.shape), dtype=value.dtype)

    def add(self, record: RLTTransition | dict[str, Any]) -> None:
        transition = record if isinstance(record, RLTTransition) else RLTTransition.from_mapping(record)
        record_np = transition.to_numpy()
        with self._lock:
            if self._storage is None:
                self._initialize_storage(record_np)
            assert self._storage is not None
            for key, value in record_np.items():
                expected_shape = self._storage[key].shape[1:]
                if value.shape != expected_shape:
                    raise ValueError(f"{key} has shape {value.shape}, expected {expected_shape}")
                self._storage[key][self._position] = value
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def extend(self, records: Iterable[RLTTransition | dict[str, Any]]) -> None:
        for record in records:
            self.add(record)

    def sample(self, batch_size: int) -> ArrayDict:
        with self._lock:
            if self._storage is None or self._size == 0:
                raise RuntimeError("Cannot sample from an empty replay buffer.")
            batch_size = min(batch_size, self._size)
            if self._sample_strategy == "stratified":
                indices = self._sample_stratified_indices(batch_size)
            elif self._sample_strategy == "uniform":
                indices = self._sample_uniform_indices(batch_size)
            else:
                raise ValueError(f"Unknown replay sample_strategy: {self._sample_strategy}")
            return {key: value[indices].copy() for key, value in self._storage.items()}

    def _sample_uniform_indices(self, batch_size: int) -> np.ndarray:
        return self._rng.integers(0, self._size, size=batch_size)

    def _sample_stratified_indices(self, batch_size: int) -> np.ndarray:
        assert self._storage is not None
        phase = self._storage["collection_phase_id"][: self._size]
        episode_id = self._storage["episode_id"][: self._size]
        source = self._storage["source"][: self._size]
        source_chunk = self._storage["source_chunk"][: self._size]
        intervention = self._storage["intervention_flag"][: self._size]
        all_indices = np.arange(self._size, dtype=np.int64)

        max_episode_id = int(np.max(episode_id)) if episode_id.size else -1
        recent_start = max_episode_id - max(self._recent_episode_window, 1) + 1
        recent_online_pool = np.flatnonzero((phase == COLLECTION_PHASE_ONLINE) & (episode_id >= recent_start))
        warmup_demo_pool = np.flatnonzero(phase == COLLECTION_PHASE_WARMUP)
        human_intervention_pool = np.flatnonzero(
            intervention
            | (source == int(TransitionSource.HUMAN))
            | (source == int(TransitionSource.MIXED))
            | np.any(source_chunk == int(TransitionSource.HUMAN), axis=1)
            | np.any(source_chunk == int(TransitionSource.MIXED), axis=1)
        )

        n_recent = int(round(batch_size * self._recent_online_ratio))
        n_warmup = int(round(batch_size * self._warmup_demo_ratio))
        n_human = int(round(batch_size * self._human_intervention_ratio))
        n_uniform = max(batch_size - n_recent - n_warmup - n_human, 0)

        indices = [
            self._sample_from_pool(recent_online_pool, n_recent),
            self._sample_from_pool(warmup_demo_pool, n_warmup),
            self._sample_from_pool(human_intervention_pool, n_human),
            self._sample_from_pool(all_indices, n_uniform),
        ]
        sampled = [part for part in indices if part.size > 0]
        result = np.concatenate(sampled) if sampled else np.empty((0,), dtype=np.int64)
        if result.size < batch_size:
            result = np.concatenate([result, self._sample_from_pool(all_indices, batch_size - result.size)])
        self._rng.shuffle(result)
        return result[:batch_size]

    def _sample_from_pool(self, pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or pool.size == 0:
            return np.empty((0,), dtype=np.int64)
        return self._rng.choice(pool, size=count, replace=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "size": self._size,
                "position": self._position,
                "sample_strategy": self._sample_strategy,
                "recent_episode_window": self._recent_episode_window,
                "recent_online_ratio": self._recent_online_ratio,
                "warmup_demo_ratio": self._warmup_demo_ratio,
                "human_intervention_ratio": self._human_intervention_ratio,
            }


class ReplayBatchSource(Protocol):
    def sample_batch(self, _batch_size: int) -> ArrayDict: ...
    def stats(self) -> dict[str, Any]: ...


class ReplayManager:
    """B3 process core object.

    It owns the CPU replay buffer, appends an on-disk journal, and optionally exposes
    itself over a lightweight local HTTP service.
    """

    def __init__(
        self,
        capacity: int,
        *,
        journal_path: str,
        seed: int = 0,
        metrics_path: str | None = None,
        sample_strategy: str = "uniform",
        recent_episode_window: int = 20,
        recent_online_ratio: float = 0.4,
        warmup_demo_ratio: float = 0.3,
        human_intervention_ratio: float = 0.2,
        tensor_contract: ReplayTensorContract | None = None,
    ):
        self._buffer = ReplayBuffer(
            capacity,
            seed=seed,
            sample_strategy=sample_strategy,
            recent_episode_window=recent_episode_window,
            recent_online_ratio=recent_online_ratio,
            warmup_demo_ratio=warmup_demo_ratio,
            human_intervention_ratio=human_intervention_ratio,
        )
        self._journal_path = journal_path
        self._tensor_contract = tensor_contract
        self._metrics_path = metrics_path
        self._lock = threading.Lock()
        self._packer = msgpack_numpy.Packer()
        self._adds_total = 0
        self._max_episode_id = -1
        os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)
        self._restore_from_journal()
        logger.info("ReplayManager initialized capacity=%s journal_path=%s", capacity, journal_path)

    def add_transition(self, transition: RLTTransition | dict[str, Any]) -> None:
        record = transition if isinstance(transition, RLTTransition) else RLTTransition.from_mapping(transition)
        self._validate_record(record, context="replay add")
        with self._lock:
            self._buffer.add(record)
            self._append_journal(record)
            self._adds_total += 1
            self._max_episode_id = max(self._max_episode_id, int(record.episode_id))
            if self._adds_total <= 5 or self._adds_total % 25 == 0:
                stats = self._buffer.stats()
                logger.info(
                    "ReplayManager size=%s/%s added_total=%s",
                    stats["size"],
                    stats["capacity"],
                    self._adds_total,
                )
                self._append_stats_metric(stats)

    def add_transitions(self, transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        records = [
            transition if isinstance(transition, RLTTransition) else RLTTransition.from_mapping(transition)
            for transition in transitions
        ]
        if not records:
            return
        for index, record in enumerate(records):
            self._validate_record(record, context=f"replay extend record {index}")

        with self._lock:
            for record in records:
                self._buffer.add(record)
                self._adds_total += 1
                self._max_episode_id = max(self._max_episode_id, int(record.episode_id))
            self._append_journal_many(records)

            stats = self._buffer.stats()
            logger.info(
                "ReplayManager size=%s/%s added_total=%s batch_size=%s",
                stats["size"],
                stats["capacity"],
                self._adds_total,
                len(records),
            )
            self._append_stats_metric(stats)

    def sample_batch(self, batch_size: int) -> ArrayDict:
        return self._buffer.sample(batch_size)

    def stats(self) -> dict[str, Any]:
        return {
            **self._buffer.stats(),
            "adds_total": self._adds_total,
            "journal_path": self._journal_path,
            "max_episode_id": self._max_episode_id,
            "tensor_contract": None if self._tensor_contract is None else dataclasses.asdict(self._tensor_contract),
        }

    def _validate_record(self, record: RLTTransition, *, context: str) -> None:
        if self._tensor_contract is not None:
            self._tensor_contract.validate(record, context=context)

    def _append_journal(self, record: RLTTransition) -> None:
        self._append_journal_many([record])

    def _append_journal_many(self, records: Iterable[RLTTransition]) -> None:
        with open(self._journal_path, "ab") as f:
            for record in records:
                pickle.dump(record.to_journal_record(), f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

    def _restore_from_journal(self) -> None:
        if not os.path.exists(self._journal_path):
            logger.info("ReplayManager journal not found at %s; starting empty.", self._journal_path)
            return
        restored = 0
        with open(self._journal_path, "rb") as f:
            while True:
                try:
                    raw = pickle.load(f)
                except EOFError:
                    break
                record = RLTTransition.from_mapping(raw)
                self._validate_record(
                    record,
                    context=f"replay journal {self._journal_path} record {restored}",
                )
                self._buffer.add(record)
                restored += 1
                self._max_episode_id = max(self._max_episode_id, int(record.episode_id))
        self._adds_total = restored
        logger.info("ReplayManager restored %s transitions from %s", restored, self._journal_path)

    def serve_forever(self, host: str, port: int, *, stop_event: threading.Event | None = None) -> None:
        manager = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self.send_response(http.HTTPStatus.OK)
                    self.end_headers()
                    self.wfile.write(b"OK\n")
                    return
                if self.path == "/stats":
                    payload = json.dumps(manager.stats()).encode("utf-8")
                    self.send_response(http.HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_error(http.HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                content_len = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_len)
                if self.path == "/add":
                    payload = msgpack_numpy.unpackb(body)
                    manager.add_transition(payload)
                    self._write_msgpack({"ok": True})
                    return
                if self.path == "/extend":
                    payload = msgpack_numpy.unpackb(body)
                    manager.add_transitions(payload["transitions"])
                    self._write_msgpack({"ok": True})
                    return
                if self.path == "/sample":
                    payload = msgpack_numpy.unpackb(body)
                    batch = manager.sample_batch(int(payload["batch_size"]))
                    self._write_msgpack(batch)
                    return
                self.send_error(http.HTTPStatus.NOT_FOUND)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _write_msgpack(self, data: Any) -> None:
                response = manager._packer.pack(data)
                self.send_response(http.HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer((host, port), Handler)
        server.timeout = 0.5
        logger.info("ReplayManager listening on http://%s:%s", host, port)
        try:
            if stop_event is None:
                server.serve_forever()
            else:
                while not stop_event.is_set():
                    server.handle_request()
        finally:
            server.server_close()
            logger.info("ReplayManager stopped.")

    def _append_stats_metric(self, stats: dict[str, Any]) -> None:
        if self._metrics_path is None:
            return
        journal_size = os.path.getsize(self._journal_path) if os.path.exists(self._journal_path) else 0
        append_jsonl(
            self._metrics_path,
            {
                "timestamp": time.time(),
                "size": stats["size"],
                "capacity": stats["capacity"],
                "position": stats["position"],
                "adds_total": self._adds_total,
                "journal_size_bytes": journal_size,
            },
        )


class ReplayClient:
    """Thin client for the local replay_manager service."""

    def __init__(self, base_url: str, *, timeout_sec: float = 1.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._packer = msgpack_numpy.Packer()

    def add_transition(self, transition: RLTTransition | dict[str, Any]) -> None:
        payload = transition.to_journal_record() if isinstance(transition, RLTTransition) else transition
        self._post("/add", payload)

    def add_transitions(self, transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        payload = []
        for transition in transitions:
            payload.append(transition.to_journal_record() if isinstance(transition, RLTTransition) else transition)
        self._post("/extend", {"transitions": payload})

    def sample_batch(self, batch_size: int) -> ArrayDict:
        return self._post("/sample", {"batch_size": batch_size})

    def stats(self) -> dict[str, Any]:
        req = urllib_request.Request(f"{self._base_url}/stats", method="GET")
        with urllib_request.urlopen(req, timeout=self._timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: Any) -> Any:
        body = self._packer.pack(payload)
        req = urllib_request.Request(
            f"{self._base_url}{path}",
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as response:
                return msgpack_numpy.unpackb(response.read())
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Replay request failed for {path}") from exc


class NullReplayClient:
    """No-op replay client used by eval-only rollout."""

    def add_transition(self, _transition: RLTTransition | dict[str, Any]) -> None:
        return

    def add_transitions(self, _transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        return

    def stats(self) -> dict[str, Any]:
        return {
            "capacity": 0,
            "size": 0,
            "position": 0,
            "adds_total": 0,
            "journal_path": None,
            "max_episode_id": -1,
        }
