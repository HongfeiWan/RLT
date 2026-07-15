# SPDX-License-Identifier: Apache-2.0

"""Offline Machine-A features for the official Nero LeRobot v3 dataset.

This module is intentionally independent from :mod:`groot_rlt.cli`. Heavy GR00T,
Transformers, Torch, and LeRobot imports happen only in the production factory
or standalone ``main``. The lightweight provider and atomic cache can therefore
be tested with a fake policy on CPU.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from groot_rlt.integration.artifact_lineage import (
    canonical_json_sha256,
    checkpoint_fingerprint,
    file_sha256,
)
from groot_rlt.integration.lerobot_v3_replay_bridge import (
    FrameIdentity,
    ReplayBridgeBundle,
    ReplayFrameFeatures,
    build_lerobot_v3_replay_bundle,
    inspect_lerobot_v3_replay_source,
    open_official_lerobot_dataset,
    write_replay_bundle,
)
from groot_rlt.integration.nero_action_contract import (
    ACTOR_PROPRIO_CHANNEL_NAMES,
    ACTOR_PROPRIO_DIM,
    ROT6D_CONVENTION,
    V3_ARM_STATE_SLICE,
    V3_EEF_STATE_SLICE,
    V3_HAND_STATE_SLICE,
    V3_STATE_DIM,
    VLA_REFERENCE_CHANNEL_NAMES,
    VLA_REFERENCE_DIM,
    bridge_v3_executed_action,
    project_v3_policy_state_to_actor_proprio,
    project_vla_reference_to_executed_action,
    semantic_layout_hash,
)
from groot_rlt.integration.prefix_cache_contract import (
    load_prefix_cache_contract,
    validate_prefix_cache_deployment_paths,
    vlm_content_fingerprint,
)

FEATURE_CACHE_SCHEMA_NAME = "groot_rlt.lerobot_v3_offline_features"
FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_TAP = "raw_backbone_pre_action_head"
PROCESSOR_MODE = "eval"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_POLICY_STATE_KEYS = ("eef_9d", "hand_joint_pos", "arm_joint_pos")
_POLICY_ACTION_KEYS = ("eef_9d", "hand_joint_target", "arm_joint_target")
_SOURCE_STATE_ORDER = "arm7+eef9+hand10"
_CHECKPOINT_STATE_ORDER = "eef9+hand10+arm7"
_RUNTIME_STATE_ORDER = "eef9+hand10"
_VLA_ACTION_HORIZON = 32
_V3_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_V3_VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_DATA_FILE_RE = re.compile(r"data/chunk-(\d{3})/file-(\d{3})\.parquet\Z")
_EPISODE_FILE_RE = re.compile(r"meta/episodes/chunk-(\d{3})/file-(\d{3})\.parquet\Z")


def _require_sha256(name: str, value: str) -> str:
    value = str(value)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must have form sha256:<64 lowercase hex>, got {value!r}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8") if isinstance(value, bytes) else value
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar, got {array.shape}")
    item = array.reshape(-1)[0].item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def _row_int(row: Mapping[str, Any], key: str) -> int:
    if key not in row:
        raise KeyError(f"official LeRobot row is missing {key!r}")
    value = _scalar(row[key], name=key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def _row_string(row: Mapping[str, Any], key: str) -> str:
    if key not in row:
        raise KeyError(f"official LeRobot row is missing {key!r}")
    value = _scalar(row[key], name=key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _fingerprint_file_set(root: Path, names: Sequence[str], *, label: str) -> str:
    root = root.expanduser().resolve()
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} at {root} is missing files: {missing}")
    hashes = {name: file_sha256(root / name) for name in sorted(names)}
    return canonical_json_sha256(hashes)


def processor_fingerprint(processor_path: str | Path) -> str:
    """Fingerprint the exact GR00T processor/statistics deployment contract."""

    root = Path(processor_path).expanduser().resolve()
    required = ["processor_config.json", "statistics.json"]
    if (root / "embodiment_id.json").is_file():
        required.append("embodiment_id.json")
    return _fingerprint_file_set(root, required, label="processor")


def dataset_content_fingerprint(
    dataset_dir: str | Path,
    *,
    camera_keys: Sequence[str],
) -> str:
    """Fingerprint all local v3 bytes consumed by offline feature generation.

    The existing replay ``dataset_fingerprint`` remains a semantic row identity.
    This independent content identity covers every metadata file, every data
    Parquet shard, and the MP4 shards for exactly the selected camera keys.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    camera_keys = tuple(camera_keys)
    if (
        not camera_keys
        or len(set(camera_keys)) != len(camera_keys)
        or not all(isinstance(key, str) and key for key in camera_keys)
    ):
        raise ValueError("camera_keys must be a non-empty unique sequence")

    required = (
        root / "meta" / "info.json",
        root / "meta" / "teleop_stack_recap.json",
        root / "meta" / "tasks.parquet",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dataset content fingerprint is missing files: {missing}")

    groups: list[tuple[str, list[Path]]] = [
        ("metadata", sorted(path for path in (root / "meta").rglob("*") if path.is_file())),
        ("data parquet", sorted((root / "data").rglob("*.parquet"))),
    ]
    for camera_key in camera_keys:
        groups.append(
            (
                f"camera {camera_key}",
                sorted((root / "videos" / camera_key).rglob("*.mp4")),
            )
        )
    for label, files in groups:
        if not files:
            raise ValueError(f"dataset content fingerprint found no {label} files under {root}")

    files = sorted({path.resolve() for _, group in groups for path in group})
    hashes = {path.relative_to(root).as_posix(): file_sha256(path) for path in files}
    return canonical_json_sha256(
        {
            "schema": "groot_rlt.lerobot_v3_consumed_content.v1",
            "camera_keys": list(camera_keys),
            "files": hashes,
        }
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CameraBinding:
    """One official v3 image feature mapped to one 400k policy video key."""

    source_key: str
    policy_key: str
    height: int
    width: int
    channels: int = 3
    source_dtype: str = "video"

    def validate(self) -> None:
        for name in ("source_key", "policy_key", "source_dtype"):
            if not getattr(self, name):
                raise ValueError(f"camera {name} must be non-empty")
        for name in ("height", "width", "channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if self.channels != 3:
            raise ValueError("Nero RGB camera contract requires exactly three channels")


def camera_contract_fingerprint(bindings: Sequence[CameraBinding]) -> str:
    """Fingerprint ordered camera aliases, shapes, and source feature types."""

    for binding in bindings:
        binding.validate()
    return canonical_json_sha256([dataclasses.asdict(binding) for binding in bindings])


def token_contract_fingerprint(
    *,
    token_scope: str,
    token_sampling: str,
    max_vl_tokens: int,
) -> str:
    """Fingerprint the exact VL-prefix selection used by the strict encoder."""

    if token_scope not in {"all", "image", "non_image"}:
        raise ValueError(f"unsupported token_scope {token_scope!r}")
    if token_sampling not in {"head", "tail", "uniform"}:
        raise ValueError(f"unsupported token_sampling {token_sampling!r}")
    if type(max_vl_tokens) is not int or max_vl_tokens <= 0:
        raise ValueError("max_vl_tokens must be a positive integer")
    return canonical_json_sha256(
        {
            "feature_tap": FEATURE_TAP,
            "processor_mode": PROCESSOR_MODE,
            "token_scope": token_scope,
            "token_sampling": token_sampling,
            "max_vl_tokens": max_vl_tokens,
        }
    )


@dataclasses.dataclass(frozen=True, slots=True)
class OfflineFeatureContract:
    """Immutable identity of one offline Machine-A feature cache."""

    dataset_fingerprint: str
    dataset_content_fingerprint: str
    checkpoint_fingerprint: str
    encoder_artifact_fingerprint: str
    processor_fingerprint: str
    prefix_cache_fingerprint: str
    vlm_deployment_content_fingerprint: str
    token_contract_fingerprint: str
    camera_contract_fingerprint: str
    model_path: str
    processor_path: str
    vlm_model_path: str
    prefix_cache_manifest_path: str
    token_scope: str
    token_sampling: str
    max_vl_tokens: int
    denoise_steps: int
    base_seed: int
    z_dim: int
    camera_bindings: tuple[CameraBinding, ...]
    action_horizon: int = _VLA_ACTION_HORIZON
    language_key: str = "annotation.human.action.task_description"
    embodiment_tag: str = "new_embodiment"
    state_dim: int = V3_STATE_DIM
    proprio_dim: int = ACTOR_PROPRIO_DIM
    reference_dim: int = VLA_REFERENCE_DIM
    rot6d_convention: str = ROT6D_CONVENTION

    def validate(self) -> None:
        for name in (
            "dataset_fingerprint",
            "dataset_content_fingerprint",
            "checkpoint_fingerprint",
            "encoder_artifact_fingerprint",
            "processor_fingerprint",
            "prefix_cache_fingerprint",
            "vlm_deployment_content_fingerprint",
            "token_contract_fingerprint",
            "camera_contract_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "model_path",
            "processor_path",
            "vlm_model_path",
            "prefix_cache_manifest_path",
        ):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute() or str(value.resolve()) != getattr(self, name):
                raise ValueError(f"{name} must be a normalized absolute deployment path")
        if self.token_contract_fingerprint != token_contract_fingerprint(
            token_scope=self.token_scope,
            token_sampling=self.token_sampling,
            max_vl_tokens=self.max_vl_tokens,
        ):
            raise ValueError("token_contract_fingerprint does not match token selection")
        if self.camera_contract_fingerprint != camera_contract_fingerprint(self.camera_bindings):
            raise ValueError("camera_contract_fingerprint does not match camera bindings")
        if type(self.denoise_steps) is not int or self.denoise_steps <= 0:
            raise ValueError("denoise_steps must be a positive integer")
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        if type(self.z_dim) is not int or self.z_dim <= 0:
            raise ValueError("z_dim must be a positive integer")
        if self.action_horizon != _VLA_ACTION_HORIZON:
            raise ValueError(
                f"400k action_horizon must be {_VLA_ACTION_HORIZON}, got {self.action_horizon}"
            )
        if (
            self.state_dim != V3_STATE_DIM
            or self.proprio_dim != ACTOR_PROPRIO_DIM
            or self.reference_dim != VLA_REFERENCE_DIM
        ):
            raise ValueError(
                "Nero offline features require source state_dim=26, proprio_dim=19, "
                "and reference_dim=26"
            )
        if self.rot6d_convention != ROT6D_CONVENTION:
            raise ValueError("offline feature rot6d convention differs from 400k inference")
        if not self.language_key or not self.embodiment_tag:
            raise ValueError("language_key and embodiment_tag must be non-empty")
        source_keys = [binding.source_key for binding in self.camera_bindings]
        policy_keys = [binding.policy_key for binding in self.camera_bindings]
        if len(source_keys) != len(set(source_keys)) or len(policy_keys) != len(set(policy_keys)):
            raise ValueError("camera source and policy keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "camera_bindings"
            },
            "camera_bindings": [dataclasses.asdict(item) for item in self.camera_bindings],
            "feature_tap": FEATURE_TAP,
            "processor_mode": PROCESSOR_MODE,
            "source_state_order": _SOURCE_STATE_ORDER,
            "checkpoint_state_order": _CHECKPOINT_STATE_ORDER,
            "runtime_state_order": _RUNTIME_STATE_ORDER,
            "runtime_proprio_layout": list(ACTOR_PROPRIO_CHANNEL_NAMES),
            "runtime_proprio_layout_hash": semantic_layout_hash(
                ACTOR_PROPRIO_CHANNEL_NAMES,
                rotation_convention=ROT6D_CONVENTION,
            ),
            "policy_state_keys": list(_POLICY_STATE_KEYS),
            "policy_action_keys": list(_POLICY_ACTION_KEYS),
            "reference_layout_hash": semantic_layout_hash(
                VLA_REFERENCE_CHANNEL_NAMES,
                rotation_convention=ROT6D_CONVENTION,
            ),
        }

    @property
    def feature_contract_fingerprint(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclasses.dataclass(frozen=True, slots=True)
class OfflineInferenceResult:
    """Validated runtime values before they are committed to the cache."""

    z_rl: np.ndarray
    vla_reference_action: np.ndarray
    proprio: np.ndarray
    original_token_count: int | None = None
    selected_token_count: int | None = None


class OfflineFeatureBackend(Protocol):
    """Inference backend implemented by the real 400k policy or a CPU fake."""

    def infer_one(
        self,
        observation: Mapping[str, Any],
        *,
        seed: int,
    ) -> OfflineInferenceResult: ...


class AtomicFeatureCache:
    """Per-frame crash-recoverable cache bound to an exact feature contract."""

    def __init__(self, root: str | Path, contract: OfflineFeatureContract):
        contract.validate()
        self.root = Path(root).expanduser().resolve()
        self.contract = contract
        self.manifest_path = self.root / "manifest.json"
        self.frames_dir = self.root / "frames"
        expected_manifest = self._expected_manifest()
        if self.manifest_path.exists():
            actual = _read_json(self.manifest_path)
            if _canonical_json(actual) != _canonical_json(expected_manifest):
                raise ValueError(
                    "feature cache manifest does not match the requested dataset/model/encoder/"
                    "processor/denoise/token/camera contract"
                )
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError(
                    f"feature cache directory is non-empty but has no manifest: {self.root}"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.manifest_path, expected_manifest)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def _expected_manifest(self) -> dict[str, Any]:
        return {
            "schema_name": FEATURE_CACHE_SCHEMA_NAME,
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "feature_contract_fingerprint": self.contract.feature_contract_fingerprint,
            "contract": self.contract.to_dict(),
            "record_format": "atomic_npz_v1",
        }

    def frame_path(self, identity: FrameIdentity) -> Path:
        return (
            self.frames_dir
            / f"episode_{identity.episode_index:06d}"
            / f"frame_{identity.frame_index:06d}.npz"
        )

    def _expected_record_metadata(
        self,
        identity: FrameIdentity,
        *,
        seed: int,
    ) -> dict[str, Any]:
        return {
            "schema_name": FEATURE_CACHE_SCHEMA_NAME,
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "feature_contract_fingerprint": self.contract.feature_contract_fingerprint,
            "dataset_fingerprint": identity.dataset_fingerprint,
            "episode_index": identity.episode_index,
            "frame_index": identity.frame_index,
            "task_episode_id": identity.task_episode_id,
            "partition": identity.partition,
            "seed": seed,
        }

    def load(
        self,
        identity: FrameIdentity,
        *,
        seed: int,
        expected_proprio: np.ndarray,
    ) -> ReplayFrameFeatures | None:
        path = self.frame_path(identity)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {
                    "metadata_json",
                    "z_rl",
                    "vla_reference_action",
                    "proprio",
                }:
                    raise ValueError(f"unexpected arrays {sorted(data.files)}")
                metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                expected_metadata = self._expected_record_metadata(identity, seed=seed)
                if _canonical_json(metadata) != _canonical_json(expected_metadata):
                    raise ValueError("record metadata does not match requested frame contract")
                z_rl = np.asarray(data["z_rl"], dtype=np.float32)
                reference = np.asarray(data["vla_reference_action"], dtype=np.float32)
                proprio = np.asarray(data["proprio"], dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"invalid cached feature frame {path}: {exc}") from exc
        self._validate_arrays(z_rl, reference, proprio, expected_proprio=expected_proprio)
        return ReplayFrameFeatures(z_rl=z_rl.copy(), vla_reference_action=reference.copy())

    def store(
        self,
        identity: FrameIdentity,
        *,
        seed: int,
        result: OfflineInferenceResult,
        expected_proprio: np.ndarray,
    ) -> ReplayFrameFeatures:
        z_rl = np.asarray(result.z_rl, dtype=np.float32)
        reference = np.asarray(result.vla_reference_action, dtype=np.float32)
        proprio = np.asarray(result.proprio, dtype=np.float32)
        self._validate_arrays(z_rl, reference, proprio, expected_proprio=expected_proprio)
        path = self.frame_path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            cached = self.load(
                identity,
                seed=seed,
                expected_proprio=expected_proprio,
            )
            assert cached is not None
            if not np.array_equal(cached.z_rl, z_rl) or not np.array_equal(
                cached.vla_reference_action, reference
            ):
                raise ValueError(f"cache race produced different deterministic features: {path}")
            return cached
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        metadata = self._expected_record_metadata(identity, seed=seed)
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    metadata_json=np.asarray(_canonical_json(metadata)),
                    z_rl=z_rl,
                    vla_reference_action=reference,
                    proprio=proprio,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return ReplayFrameFeatures(z_rl=z_rl.copy(), vla_reference_action=reference.copy())

    def _validate_arrays(
        self,
        z_rl: np.ndarray,
        reference: np.ndarray,
        proprio: np.ndarray,
        *,
        expected_proprio: np.ndarray,
    ) -> None:
        if z_rl.shape != (self.contract.z_dim,) or not np.isfinite(z_rl).all():
            raise ValueError(
                f"z_rl must be finite shape ({self.contract.z_dim},), got {z_rl.shape}"
            )
        if reference.shape != (VLA_REFERENCE_DIM,) or not np.isfinite(reference).all():
            raise ValueError(
                f"vla_reference_action must be finite shape ({VLA_REFERENCE_DIM},), "
                f"got {reference.shape}"
            )
        if proprio.shape != (ACTOR_PROPRIO_DIM,) or not np.isfinite(proprio).all():
            raise ValueError(
                f"proprio must be finite shape ({ACTOR_PROPRIO_DIM},), got {proprio.shape}"
            )
        expected = np.asarray(expected_proprio, dtype=np.float32)
        if not np.array_equal(proprio, expected):
            raise ValueError("backend/cache proprio does not exactly match eef9+hand10")
        projected = project_vla_reference_to_executed_action(reference)
        bridge_v3_executed_action(projected, rotation_convention=ROT6D_CONVENTION)

    def summary(self) -> dict[str, Any]:
        files = list(self.frames_dir.glob("episode_*/frame_*.npz"))
        temporary = list(self.root.rglob("*.tmp"))
        return {
            "root": str(self.root),
            "feature_contract_fingerprint": self.contract.feature_contract_fingerprint,
            "dataset_content_fingerprint": self.contract.dataset_content_fingerprint,
            "completed_frame_count": len(files),
            "temporary_file_count": len(temporary),
        }


def frame_seed(identity: FrameIdentity, *, base_seed: int) -> int:
    """Derive a stable Torch/NumPy/Python seed from one immutable frame key."""

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    material = (
        f"{base_seed}\0{identity.dataset_fingerprint}\0{identity.episode_index}\0"
        f"{identity.frame_index}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big")
    return value % (2**63 - 1)


def _normalize_rgb_image(value: Any, binding: CameraBinding) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    expected_hwc = (binding.height, binding.width, binding.channels)
    expected_chw = (binding.channels, binding.height, binding.width)
    if array.shape == expected_chw:
        array = np.transpose(array, (1, 2, 0))
    elif array.shape != expected_hwc:
        raise ValueError(
            f"camera {binding.source_key!r} has shape {array.shape}, expected "
            f"HWC {expected_hwc} or CHW {expected_chw}"
        )
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(f"camera {binding.source_key!r} contains non-finite pixels")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum >= 0.0 and maximum <= 1.0 + 1.0e-6:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
        elif minimum >= 0.0 and maximum <= 255.0:
            array = np.rint(array)
        else:
            raise ValueError(
                f"camera {binding.source_key!r} floating pixels must be in [0,1] or [0,255]"
            )
    elif np.issubdtype(array.dtype, np.integer):
        if int(array.min()) < 0 or int(array.max()) > 255:
            raise ValueError(f"camera {binding.source_key!r} integer pixels must be in [0,255]")
    else:
        raise ValueError(f"camera {binding.source_key!r} has unsupported dtype {array.dtype}")
    return np.asarray(array, dtype=np.uint8)


@dataclasses.dataclass(frozen=True, slots=True)
class _LocalFrameRecord:
    row: Mapping[str, Any]
    file_key: tuple[int, int]
    file_frame_index: int


class _SequentialPyAVReader:
    """Strict, lazy, sequential RGB decoder for one local v3 video shard."""

    def __init__(
        self,
        *,
        av_module: Any,
        path: Path,
        frame_count: int,
        fps: float,
        shape: tuple[int, int, int],
    ):
        self._av = av_module
        self.path = path
        self.frame_count = frame_count
        self.fps = fps
        self.shape = shape
        self._container: Any | None = None
        self._iterator: Any | None = None
        self._next_index = 0
        self._validate_entire_video()

    def _open(self) -> tuple[Any, Any]:
        container = self._av.open(str(self.path), mode="r")
        video_streams = list(container.streams.video)
        if len(video_streams) != 1:
            container.close()
            raise ValueError(f"video shard must have exactly one video stream: {self.path}")
        if list(container.streams.audio):
            container.close()
            raise ValueError(f"video shard unexpectedly contains audio: {self.path}")
        stream = video_streams[0]
        height, width, _ = self.shape
        if (int(stream.height), int(stream.width)) != (height, width):
            container.close()
            raise ValueError(
                f"video dimensions for {self.path} are "
                f"{stream.height}x{stream.width}, expected {height}x{width}"
            )
        rate = stream.average_rate
        if rate is None or not np.isclose(float(rate), self.fps, rtol=1.0e-6, atol=1.0e-6):
            container.close()
            raise ValueError(f"video fps for {self.path} is {rate}, expected {self.fps}")
        return container, stream

    def _rgb(self, frame: Any, *, frame_index: int) -> np.ndarray:
        array = np.asarray(frame.to_ndarray(format="rgb24"))
        if array.dtype != np.uint8 or array.shape != self.shape:
            raise ValueError(
                f"decoded frame {frame_index} from {self.path} has "
                f"dtype/shape {array.dtype}/{array.shape}, expected uint8/{self.shape}"
            )
        return array

    def _validate_entire_video(self) -> None:
        container, stream = self._open()
        count = 0
        try:
            for count, frame in enumerate(container.decode(stream), start=1):
                self._rgb(frame, frame_index=count - 1)
        finally:
            container.close()
        if count != self.frame_count:
            raise ValueError(
                f"video frame count for {self.path} is {count}, expected {self.frame_count}"
            )

    def _reset(self) -> None:
        self.close()
        self._container, stream = self._open()
        self._iterator = iter(self._container.decode(stream))
        self._next_index = 0

    def get(self, frame_index: int) -> np.ndarray:
        if type(frame_index) is not int or not 0 <= frame_index < self.frame_count:
            raise IndexError(frame_index)
        if self._iterator is None or frame_index < self._next_index:
            self._reset()
        assert self._iterator is not None
        while self._next_index <= frame_index:
            try:
                frame = next(self._iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"video {self.path} ended before validated frame {frame_index}"
                ) from exc
            current = self._next_index
            self._next_index += 1
        return self._rgb(frame, frame_index=current).copy()

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._iterator = None
        self._next_index = 0

    def __del__(self) -> None:
        self.close()


@dataclasses.dataclass(frozen=True, slots=True)
class _LazyPyAVFrame:
    reader: _SequentialPyAVReader
    frame_index: int

    def numpy(self) -> np.ndarray:
        return self.reader.get(self.frame_index)


class LocalV3ParquetPyAVLoader:
    """Strict local official-v3 reader used when LeRobot is unavailable.

    Pandas, PyArrow, and PyAV are imported only when this fallback is selected.
    Every parquet row and every video frame is validated during construction;
    RGB pixels remain lazy so already cached Machine-A frames do not decode a
    second time.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        info_payload: Mapping[str, Any],
        camera_keys: Sequence[str],
        load_videos: bool = True,
    ):
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("local LeRobot v3 fallback requires pandas and PyArrow") from exc
        av = None
        if load_videos:
            try:
                import av
            except ImportError as exc:
                raise RuntimeError("local LeRobot v3 video loading requires PyAV") from exc

        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        self.info = dict(info_payload)
        self._pd = pd
        self._av = av
        self._load_videos = bool(load_videos)
        self.camera_keys = tuple(camera_keys)
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be a non-empty unique sequence")
        self._validate_info()
        self._records, file_row_counts = self._load_data_files()
        self._tasks = self._load_episode_tasks()
        self._readers = (
            self._load_and_validate_videos(file_row_counts) if self._load_videos else {}
        )
        self.num_frames = len(self._records)
        self.num_episodes = int(self.info["total_episodes"])

    @staticmethod
    def _strict_positive_int(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def _validate_info(self) -> None:
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("local fallback requires codebase_version='v3.0'")
        if self.info.get("data_path") != _V3_DATA_PATH_TEMPLATE:
            raise ValueError(f"unsupported info.data_path template {self.info.get('data_path')!r}")
        if self.info.get("video_path") != _V3_VIDEO_PATH_TEMPLATE:
            raise ValueError(
                f"unsupported info.video_path template {self.info.get('video_path')!r}"
            )
        self._strict_positive_int(self.info.get("total_frames"), name="total_frames")
        self._strict_positive_int(self.info.get("total_episodes"), name="total_episodes")
        self._strict_positive_int(self.info.get("total_tasks"), name="total_tasks")
        fps = self.info.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0:
            raise ValueError("info.fps must be a positive number")
        self.fps = float(fps)
        features = self.info.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("info.features must be an object")
        for key in self.camera_keys:
            feature = features.get(key)
            if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
                raise ValueError(f"info feature {key!r} must be a video")
            shape = feature.get("shape")
            if (
                not isinstance(shape, Sequence)
                or isinstance(shape, (str, bytes))
                or len(shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
                or any(int(value) <= 0 for value in shape)
                or int(shape[2]) != 3
            ):
                raise ValueError(f"info feature {key!r} must have RGB shape [H,W,3]")
            video_info = feature.get("info")
            if not isinstance(video_info, Mapping):
                raise ValueError(f"info feature {key!r} is missing video metadata")
            expected = {
                "video.height": int(shape[0]),
                "video.width": int(shape[1]),
                "video.channels": 3,
                "video.fps": self.fps,
                "video.is_depth_map": False,
                "has_audio": False,
            }
            for name, value in expected.items():
                actual = video_info.get(name)
                if isinstance(value, float):
                    valid = isinstance(actual, (int, float)) and np.isclose(
                        float(actual), value, rtol=1.0e-6, atol=1.0e-6
                    )
                else:
                    valid = type(actual) is type(value) and actual == value
                if not valid:
                    raise ValueError(f"info feature {key!r} {name}={actual!r}, expected {value!r}")

    def _matching_files(
        self,
        directory: Path,
        pattern: re.Pattern[str],
        *,
        label: str,
    ) -> list[tuple[tuple[int, int], Path]]:
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {directory}")
        matched: list[tuple[tuple[int, int], Path]] = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(self.root).as_posix()
            match = pattern.fullmatch(relative)
            if match is None:
                raise ValueError(f"unexpected {label} file path: {relative}")
            matched.append(((int(match.group(1)), int(match.group(2))), path))
        if not matched:
            raise ValueError(f"no {label} files found under {directory}")
        keys = [key for key, _ in matched]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate {label} chunk/file indices")
        return sorted(matched)

    def _load_data_files(
        self,
    ) -> tuple[list[_LocalFrameRecord], dict[tuple[int, int], int]]:
        files = self._matching_files(self.root / "data", _DATA_FILE_RE, label="data parquet")
        features = self.info["features"]
        expected_columns = {
            key
            for key, feature in features.items()
            if not isinstance(feature, Mapping) or feature.get("dtype") != "video"
        }
        records: list[_LocalFrameRecord] = []
        file_row_counts: dict[tuple[int, int], int] = {}
        for file_key, path in files:
            table = self._pd.read_parquet(path)
            actual_columns = set(table.columns)
            if actual_columns != expected_columns:
                missing = sorted(expected_columns - actual_columns)
                extra = sorted(actual_columns - expected_columns)
                raise ValueError(
                    f"data schema mismatch in {path}: missing={missing}, extra={extra}"
                )
            rows = table.to_dict(orient="records")
            if not rows:
                raise ValueError(f"empty data parquet shard: {path}")
            file_row_counts[file_key] = len(rows)
            records.extend(
                _LocalFrameRecord(row=row, file_key=file_key, file_frame_index=index)
                for index, row in enumerate(rows)
            )
        total_frames = int(self.info["total_frames"])
        if len(records) != total_frames:
            raise ValueError(f"data parquet row count is {len(records)}, expected {total_frames}")
        self._validate_data_rows(records)
        return records, file_row_counts

    def _validate_data_rows(self, records: Sequence[_LocalFrameRecord]) -> None:
        total_episodes = int(self.info["total_episodes"])
        episode_frames: dict[int, list[int]] = {}
        for position, record in enumerate(records):
            row = record.row
            if _row_int(row, "index") != position:
                raise ValueError(
                    f"data row index at position {position} is not globally contiguous"
                )
            episode_index = _row_int(row, "episode_index")
            frame_index = _row_int(row, "frame_index")
            if episode_index >= total_episodes:
                raise ValueError(f"data row {position} has invalid episode_index")
            episode_frames.setdefault(episode_index, []).append(frame_index)
            timestamp = float(_scalar(row["timestamp"], name="timestamp"))
            expected_timestamp = frame_index / self.fps
            if not np.isfinite(timestamp) or not np.isclose(
                timestamp, expected_timestamp, rtol=1.0e-5, atol=1.0e-5
            ):
                raise ValueError(
                    f"data row {position} timestamp {timestamp} does not match "
                    f"frame_index/fps {expected_timestamp}"
                )
        if set(episode_frames) != set(range(total_episodes)):
            raise ValueError("data episode_index values must be contiguous from zero")
        for episode_index, frame_indices in episode_frames.items():
            if frame_indices != list(range(len(frame_indices))):
                raise ValueError(
                    f"episode {episode_index} frame_index values are not contiguous from zero"
                )

    @staticmethod
    def _task_strings(value: Any, *, episode_index: int) -> tuple[str, ...]:
        array = _to_numpy(value).reshape(-1)
        items = array.tolist()
        if not items or any(not isinstance(item, str) or not item for item in items):
            raise ValueError(f"episode {episode_index} has no non-empty task string")
        return tuple(items)

    def _load_episode_tasks(self) -> dict[int, str]:
        episode_files = self._matching_files(
            self.root / "meta" / "episodes",
            _EPISODE_FILE_RE,
            label="episode parquet",
        )
        episode_rows: list[Mapping[str, Any]] = []
        for _, path in episode_files:
            episode_rows.extend(self._pd.read_parquet(path).to_dict(orient="records"))
        total_episodes = int(self.info["total_episodes"])
        if len(episode_rows) != total_episodes:
            raise ValueError(
                f"episode metadata row count is {len(episode_rows)}, expected {total_episodes}"
            )

        tasks_path = self.root / "meta" / "tasks.parquet"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"tasks metadata not found: {tasks_path}")
        tasks_table = self._pd.read_parquet(tasks_path)
        if "task_index" not in tasks_table.columns:
            raise ValueError("meta/tasks.parquet is missing task_index")
        task_indices = []
        for position, value in enumerate(tasks_table["task_index"].tolist()):
            value = _scalar(value, name=f"tasks[{position}].task_index")
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("meta/tasks.parquet task_index values must be integers")
            task_indices.append(int(value))
        total_tasks = int(self.info["total_tasks"])
        if sorted(task_indices) != list(range(total_tasks)) or len(set(task_indices)) != len(
            task_indices
        ):
            raise ValueError("meta/tasks.parquet task_index values must be contiguous from zero")
        task_by_index: dict[int, str] = {}
        if "task" in tasks_table.columns:
            for row in tasks_table.to_dict(orient="records"):
                task = row["task"]
                if not isinstance(task, str) or not task:
                    raise ValueError("meta/tasks.parquet task values must be non-empty strings")
                task_by_index[int(row["task_index"])] = task

        by_episode: dict[int, Mapping[str, Any]] = {}
        for row in episode_rows:
            episode_index = _row_int(row, "episode_index")
            if episode_index in by_episode:
                raise ValueError(f"duplicate episode metadata for {episode_index}")
            by_episode[episode_index] = row
        if set(by_episode) != set(range(total_episodes)):
            raise ValueError("episode metadata indices must be contiguous from zero")

        result: dict[int, str] = {}
        for episode_index, metadata in sorted(by_episode.items()):
            length = _row_int(metadata, "length")
            start = _row_int(metadata, "dataset_from_index")
            stop = _row_int(metadata, "dataset_to_index")
            if stop - start != length or not 0 <= start < stop <= len(self._records):
                raise ValueError(f"episode {episode_index} has invalid dataset index range")
            records = self._records[start:stop]
            if any(_row_int(record.row, "episode_index") != episode_index for record in records):
                raise ValueError(f"episode {episode_index} metadata range selects other episodes")
            if len(records) != length:
                raise ValueError(f"episode {episode_index} length metadata is inconsistent")
            data_key = (
                _row_int(metadata, "data/chunk_index"),
                _row_int(metadata, "data/file_index"),
            )
            if any(record.file_key != data_key for record in records):
                raise ValueError(f"episode {episode_index} crosses its declared data shard")
            indices = {_row_int(record.row, "task_index") for record in records}
            if len(indices) != 1:
                raise ValueError(f"episode {episode_index} has multiple task_index values")
            task_index = indices.pop()
            if task_index not in task_indices:
                raise ValueError(f"episode {episode_index} has unknown task_index {task_index}")
            episode_tasks = self._task_strings(metadata.get("tasks"), episode_index=episode_index)
            if len(episode_tasks) != 1:
                raise ValueError(f"episode {episode_index} must resolve to exactly one task string")
            task = episode_tasks[0]
            known = task_by_index.setdefault(task_index, task)
            if known != task:
                raise ValueError(f"task_index {task_index} maps to conflicting task strings")
            result[episode_index] = task
            self._validate_episode_video_metadata(
                metadata,
                episode_index=episode_index,
                records=records,
            )
        if set(task_by_index) != set(range(total_tasks)):
            raise ValueError("not every task_index resolves to an exact task string")
        return result

    def _validate_episode_video_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        episode_index: int,
        records: Sequence[_LocalFrameRecord],
    ) -> None:
        expected_from = records[0].file_frame_index / self.fps
        expected_to = (records[-1].file_frame_index + 1) / self.fps
        data_key = (
            _row_int(metadata, "data/chunk_index"),
            _row_int(metadata, "data/file_index"),
        )
        for key in self.camera_keys:
            prefix = f"videos/{key}"
            video_key = (
                _row_int(metadata, f"{prefix}/chunk_index"),
                _row_int(metadata, f"{prefix}/file_index"),
            )
            if video_key != data_key:
                raise ValueError(
                    f"episode {episode_index} video {key!r} does not use its data shard"
                )
            for suffix, expected in (
                ("from_timestamp", expected_from),
                ("to_timestamp", expected_to),
            ):
                name = f"{prefix}/{suffix}"
                value = float(_scalar(metadata[name], name=name))
                if not np.isfinite(value) or not np.isclose(
                    value, expected, rtol=1.0e-5, atol=1.0e-5
                ):
                    raise ValueError(f"episode {episode_index} {name}={value}, expected {expected}")

    def _load_and_validate_videos(
        self,
        file_row_counts: Mapping[tuple[int, int], int],
    ) -> dict[tuple[str, tuple[int, int]], _SequentialPyAVReader]:
        readers: dict[tuple[str, tuple[int, int]], _SequentialPyAVReader] = {}
        expected_file_keys = set(file_row_counts)
        for camera_key in self.camera_keys:
            prefix = re.escape(f"videos/{camera_key}")
            pattern = re.compile(rf"{prefix}/chunk-(\d{{3}})/file-(\d{{3}})\.mp4\Z")
            files = self._matching_files(
                self.root / "videos" / camera_key,
                pattern,
                label=f"video {camera_key}",
            )
            actual_file_keys = {file_key for file_key, _ in files}
            if actual_file_keys != expected_file_keys:
                raise ValueError(
                    f"video shards for {camera_key!r} do not match data shards: "
                    f"expected={sorted(expected_file_keys)}, got={sorted(actual_file_keys)}"
                )
            shape = tuple(int(value) for value in self.info["features"][camera_key]["shape"])
            for file_key, path in files:
                readers[(camera_key, file_key)] = _SequentialPyAVReader(
                    av_module=self._av,
                    path=path,
                    frame_count=file_row_counts[file_key],
                    fps=self.fps,
                    shape=shape,
                )
        return readers

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int:
            raise TypeError("local v3 row index must be an integer")
        if index < 0:
            index += len(self._records)
        if not 0 <= index < len(self._records):
            raise IndexError(index)
        record = self._records[index]
        row = dict(record.row)
        episode_index = _row_int(row, "episode_index")
        row["task"] = self._tasks[episode_index]
        if self._load_videos:
            for key in self.camera_keys:
                row[key] = _LazyPyAVFrame(
                    self._readers[(key, record.file_key)], record.file_frame_index
                )
        return row

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()


def open_v3_dataset_loader(
    *,
    repo_id: str,
    root: str | Path,
    info_payload: Mapping[str, Any],
    camera_keys: Sequence[str],
    loader_mode: str = "auto",
    video_backend: str = "pyav",
    dataset_factory: Callable[..., Any] | None = None,
) -> Any:
    """Prefer the official loader and fail over only when LeRobot is absent."""

    if loader_mode not in {"auto", "official", "local"}:
        raise ValueError("loader_mode must be 'auto', 'official', or 'local'")
    if loader_mode == "local":
        if dataset_factory is not None:
            raise ValueError("dataset_factory cannot be used with loader_mode='local'")
        return LocalV3ParquetPyAVLoader(
            root,
            info_payload=info_payload,
            camera_keys=camera_keys,
        )
    try:
        return open_official_lerobot_dataset(
            repo_id=repo_id,
            root=Path(root).expanduser().resolve(),
            download_videos=False,
            video_backend=video_backend,
            dataset_factory=dataset_factory,
        )
    except RuntimeError as exc:
        missing_lerobot = isinstance(exc.__cause__, ImportError) and str(exc).startswith(
            "LeRobot is not installed."
        )
        if loader_mode == "official" or not missing_lerobot:
            raise
    return LocalV3ParquetPyAVLoader(
        root,
        info_payload=info_payload,
        camera_keys=camera_keys,
    )


class OfflineV3MachineAFeatureProvider:
    """Bridge-compatible provider backed by 400k inference and an atomic cache."""

    def __init__(
        self,
        backend: OfflineFeatureBackend,
        contract: OfflineFeatureContract,
        cache: AtomicFeatureCache,
        *,
        instruction_override: str | None = None,
        task_key: str = "task",
    ):
        contract.validate()
        if cache.contract.feature_contract_fingerprint != contract.feature_contract_fingerprint:
            raise ValueError("provider and cache feature contracts differ")
        if instruction_override is not None and not instruction_override.strip():
            raise ValueError("instruction_override must be non-empty when provided")
        if not task_key:
            raise ValueError("task_key must be non-empty")
        self.backend = backend
        self.contract = contract
        self.cache = cache
        self.instruction_override = instruction_override
        self.task_key = task_key
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def feature_contract_fingerprint(self) -> str:
        return self.contract.feature_contract_fingerprint

    def __call__(
        self,
        identity: FrameIdentity,
        row: Mapping[str, Any],
    ) -> ReplayFrameFeatures:
        if identity.dataset_fingerprint != self.contract.dataset_fingerprint:
            raise ValueError(
                "frame dataset fingerprint does not match offline feature contract: "
                f"{identity.dataset_fingerprint} != {self.contract.dataset_fingerprint}"
            )
        episode_index = _row_int(row, "episode_index")
        frame_index = _row_int(row, "frame_index")
        if (episode_index, frame_index) != (identity.episode_index, identity.frame_index):
            raise ValueError("row episode/frame does not match FrameIdentity")
        if _row_string(row, "teleop_stack.task_episode_id") != identity.task_episode_id:
            raise ValueError("row task_episode_id does not match FrameIdentity")
        if _row_string(row, "teleop_stack.partition") != identity.partition:
            raise ValueError("row partition does not match FrameIdentity")

        source_state = np.asarray(_to_numpy(row["observation.state"]), dtype=np.float32)
        expected_proprio = project_v3_policy_state_to_actor_proprio(
            source_state,
            rotation_convention=ROT6D_CONVENTION,
        )
        seed = frame_seed(identity, base_seed=self.contract.base_seed)
        cached = self.cache.load(
            identity,
            seed=seed,
            expected_proprio=expected_proprio,
        )
        if cached is not None:
            self.cache_hits += 1
            return cached

        observation = self._observation(row, source_state=source_state)
        result = self.backend.infer_one(observation, seed=seed)
        self.cache_misses += 1
        return self.cache.store(
            identity,
            seed=seed,
            result=result,
            expected_proprio=expected_proprio,
        )

    def _observation(
        self,
        row: Mapping[str, Any],
        *,
        source_state: np.ndarray,
    ) -> dict[str, Any]:
        if source_state.shape != (V3_STATE_DIM,):
            raise ValueError(f"observation.state must have shape ({V3_STATE_DIM},)")
        videos = {}
        for binding in self.contract.camera_bindings:
            if binding.source_key not in row:
                raise KeyError(f"official LeRobot row is missing camera {binding.source_key!r}")
            image = _normalize_rgb_image(row[binding.source_key], binding)
            videos[binding.policy_key] = image[None, None]
        states = {
            "eef_9d": source_state[V3_EEF_STATE_SLICE][None, None].copy(),
            "hand_joint_pos": source_state[V3_HAND_STATE_SLICE][None, None].copy(),
            "arm_joint_pos": source_state[V3_ARM_STATE_SLICE][None, None].copy(),
        }
        instruction = self.instruction_override
        if instruction is None:
            instruction = _row_string(row, self.task_key)
        return {
            "video": videos,
            "state": states,
            "language": {self.contract.language_key: [[instruction]]},
        }


@contextlib.contextmanager
def _deterministic_rng(torch_module: Any, *, seed: int, device: Any):
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    devices: list[int] = []
    if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda":
        device_index = getattr(device, "index", None)
        devices = [torch_module.cuda.current_device() if device_index is None else device_index]
    try:
        with torch_module.random.fork_rng(devices=devices, enabled=True):
            torch_module.manual_seed(seed)
            if devices:
                torch_module.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class GrootOfflineMachineABackend:
    """One-pass strict encoder plus frozen 400k action-reference inference."""

    def __init__(
        self,
        *,
        policy: Any,
        encoder: Any,
        pack_vl_tokens: Callable[..., Any],
        torch_module: Any,
        device: Any,
        contract: OfflineFeatureContract,
        processor_message_factory: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    ):
        contract.validate()
        self.policy = policy
        self.encoder = encoder
        self.pack_vl_tokens = pack_vl_tokens
        self.torch = torch_module
        self.device = torch_module.device(device)
        self.contract = contract
        self.processor_message_factory = processor_message_factory
        self._validate_policy_contract()

    def _validate_policy_contract(self) -> None:
        modalities = self.policy.modality_configs
        video = modalities["video"]
        state = modalities["state"]
        action = modalities["action"]
        language = modalities["language"]
        expected_video = tuple(binding.policy_key for binding in self.contract.camera_bindings)
        if tuple(video.modality_keys) != expected_video:
            raise ValueError(
                f"400k video keys {tuple(video.modality_keys)} do not match {expected_video}"
            )
        if tuple(state.modality_keys) != _POLICY_STATE_KEYS:
            raise ValueError(
                f"400k state keys {tuple(state.modality_keys)} do not match {_POLICY_STATE_KEYS}"
            )
        if tuple(action.modality_keys) != _POLICY_ACTION_KEYS:
            raise ValueError(
                f"400k action keys {tuple(action.modality_keys)} do not match {_POLICY_ACTION_KEYS}"
            )
        if tuple(language.modality_keys) != (self.contract.language_key,):
            raise ValueError("400k language key does not match offline feature contract")
        if (
            list(video.delta_indices) != [0]
            or list(state.delta_indices) != [0]
            or list(language.delta_indices) != [0]
        ):
            raise ValueError(
                "offline row inference requires 400k video/state/language delta_indices == [0]"
            )
        if list(action.delta_indices) != list(range(self.contract.action_horizon)):
            raise ValueError(
                "400k action delta_indices must be contiguous over the exact 32-step horizon"
            )
        if self.policy.language_key != self.contract.language_key:
            raise ValueError("policy.language_key differs from the processor modality contract")
        encoder_config = self.encoder.config
        if int(encoder_config.rl_token_dim) != self.contract.z_dim:
            raise ValueError("strict encoder z dimension differs from feature contract")
        if self.contract.max_vl_tokens > int(encoder_config.max_vl_tokens):
            raise ValueError("token selection exceeds strict encoder max_vl_tokens")
        backbone_dim = getattr(self.policy.model.config, "backbone_embedding_dim", None)
        if backbone_dim is not None and int(backbone_dim) != int(encoder_config.input_dim):
            raise ValueError("400k backbone dimension differs from strict encoder input dimension")
        missing = [
            name
            for name in ("prepare_input", "backbone", "action_head")
            if not hasattr(self.policy.model, name)
        ]
        head = getattr(self.policy.model, "action_head", None)
        for name in ("_encode_features", "get_action_with_features"):
            if head is None or not hasattr(head, name):
                missing.append(f"action_head.{name}")
        if missing:
            raise ValueError(f"400k model lacks required offline Machine-A APIs: {missing}")
        head.num_inference_timesteps = self.contract.denoise_steps
        if int(head.num_inference_timesteps) != self.contract.denoise_steps:
            raise ValueError("failed to set exact denoise_steps on 400k action head")

    def _processed_input(self, item: Mapping[str, Any]) -> Any:
        if self.processor_message_factory is not None:
            return self.processor_message_factory(self.policy, item)
        from gr00t.data.types import MessageType, VLAStepData

        step = VLAStepData(
            images=item["video"],
            states=item["state"],
            actions={},
            text=item["language"][self.policy.language_key][0],
            embodiment=self.policy.embodiment_tag,
        )
        return self.policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])

    def _prepare_policy_inputs(
        self, observation: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
        if self.policy.strict:
            self.policy.check_observation(observation)
        unbatched = self.policy._unbatch_observation(observation)
        if len(unbatched) != 1:
            raise ValueError("offline Machine-A inference requires exactly one observation")
        processed = [self._processed_input(unbatched[0])]
        collated = self.policy.collate_fn(processed)
        collated = self.policy._rec_to_dtype(collated, dtype=self.torch.bfloat16)
        return collated, [unbatched[0]["state"]]

    def infer_one(
        self,
        observation: Mapping[str, Any],
        *,
        seed: int,
    ) -> OfflineInferenceResult:
        with _deterministic_rng(self.torch, seed=seed, device=self.device):
            collated, states = self._prepare_policy_inputs(observation)
            with self.torch.inference_mode():
                backbone_inputs, action_inputs = self.policy.model.prepare_input(**collated)
                backbone_output = self.policy.model.backbone(backbone_inputs)
                packed, packed_mask, _, original_counts, selected_counts = self.pack_vl_tokens(
                    backbone_output,
                    token_scope=self.contract.token_scope,
                    max_tokens=self.contract.max_vl_tokens,
                    token_sampling=self.contract.token_sampling,
                )
                features = self.policy.model.action_head._encode_features(
                    backbone_output, action_inputs
                )
                model_pred = self.policy.model.action_head.get_action_with_features(
                    backbone_features=features.backbone_features,
                    state_features=features.state_features,
                    embodiment_id=action_inputs.embodiment_id,
                    backbone_output=backbone_output,
                    action_input=action_inputs,
                    options=None,
                )
                z_rl = self.encoder.encode_rl_token(packed.float(), packed_mask)

        normalized_action = model_pred["action_pred"].float().cpu().numpy()
        batched_states = {
            key: np.stack([state[key] for state in states], axis=0) for key in _POLICY_STATE_KEYS
        }
        decoded = self.policy.processor.decode_action(
            normalized_action,
            self.policy.embodiment_tag,
            batched_states,
        )
        parts = [np.asarray(decoded[key][0], dtype=np.float32) for key in _POLICY_ACTION_KEYS]
        if any(part.ndim != 2 for part in parts):
            raise ValueError("decoded 400k actions must have [horizon, dimension] per key")
        reference_chunk = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
        if reference_chunk.ndim != 2 or reference_chunk.shape[1] != VLA_REFERENCE_DIM:
            raise ValueError(
                f"decoded 400k action must have shape [H, {VLA_REFERENCE_DIM}], "
                f"got {reference_chunk.shape}"
            )
        if reference_chunk.shape[0] != self.contract.action_horizon:
            raise ValueError(
                f"decoded 400k action horizon must be {self.contract.action_horizon}, "
                f"got {reference_chunk.shape[0]}"
            )
        if not np.isfinite(reference_chunk).all():
            raise ValueError("decoded 400k action chunk is non-finite")
        proprio_parts = []
        for key in _POLICY_STATE_KEYS[:2]:
            value = np.asarray(states[0][key], dtype=np.float32)
            if value.ndim == 2:
                value = value[-1]
            if value.ndim != 1:
                raise ValueError(f"unbatched state[{key!r}] has invalid shape {value.shape}")
            proprio_parts.append(value)
        proprio = np.concatenate(proprio_parts).astype(np.float32, copy=False)
        if proprio.shape != (ACTOR_PROPRIO_DIM,):
            raise ValueError(
                f"actor/critic proprio must have shape ({ACTOR_PROPRIO_DIM},), "
                f"got {proprio.shape}"
            )
        z_array = np.asarray(z_rl[0].detach().float().cpu().numpy(), dtype=np.float32)
        return OfflineInferenceResult(
            z_rl=z_array,
            vla_reference_action=reference_chunk[0].copy(),
            proprio=proprio.copy(),
            original_token_count=int(original_counts[0]),
            selected_token_count=int(selected_counts[0]),
        )


def _camera_bindings_from_info(
    info: Mapping[str, Any],
    *,
    source_ego_key: str,
    source_wrist_key: str,
) -> tuple[CameraBinding, ...]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.features must be an object")
    result = []
    for source_key, policy_key in (
        (source_ego_key, "ego_view"),
        (source_wrist_key, "wrist_view"),
    ):
        feature = features.get(source_key)
        if not isinstance(feature, Mapping):
            raise ValueError(f"info.features.{source_key} must be an object")
        shape = feature.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 3:
            raise ValueError(f"info.features.{source_key}.shape must be [H,W,3]")
        if feature.get("dtype") != "video" or list(shape)[2] != 3:
            raise ValueError(f"info.features.{source_key} must be RGB video")
        result.append(
            CameraBinding(
                source_key=source_key,
                policy_key=policy_key,
                height=int(shape[0]),
                width=int(shape[1]),
                channels=int(shape[2]),
                source_dtype="video",
            )
        )
    return tuple(result)


def inspect_deployment_contract(
    *,
    dataset_dir: str | Path,
    model_path: str | Path,
    processor_path: str | Path,
    vlm_model_path: str | Path,
    encoder_artifact: str | Path,
    prefix_cache_manifest: str | Path,
    info_payload: Mapping[str, Any],
    source_ego_key: str,
    source_wrist_key: str,
    token_scope: str,
    token_sampling: str,
    max_vl_tokens: int,
) -> dict[str, Any]:
    """Compute file/token/camera identities without importing Torch or GR00T."""

    cache_contract = load_prefix_cache_contract(prefix_cache_manifest)
    requested_token_contract = {
        "token_scope": token_scope,
        "token_sampling": token_sampling,
        "max_vl_tokens": max_vl_tokens,
    }
    signed_token_contract = {
        "token_scope": cache_contract.token_scope,
        "token_sampling": cache_contract.token_sampling,
        "max_vl_tokens": cache_contract.max_vl_tokens,
    }
    if requested_token_contract != signed_token_contract:
        raise ValueError(
            "Offline token selection must exactly match the signed prefix-cache contract: "
            f"requested={requested_token_contract!r} signed={signed_token_contract!r}"
        )
    model_path, processor_path, vlm_model_path = validate_prefix_cache_deployment_paths(
        cache_contract,
        model_path=model_path,
        processor_path=processor_path,
        vlm_model_path=vlm_model_path,
        context="Offline",
    )
    checkpoint_sha, _ = checkpoint_fingerprint(model_path)
    if checkpoint_sha != cache_contract.checkpoint_fingerprint:
        raise ValueError(
            "Offline 400k fingerprint differs from signed prefix-cache lineage: "
            f"model={checkpoint_sha} cache={cache_contract.checkpoint_fingerprint}"
        )
    encoder_path = Path(encoder_artifact).expanduser().resolve()
    if not encoder_path.is_file():
        raise FileNotFoundError(f"encoder artifact not found: {encoder_path}")
    cameras = _camera_bindings_from_info(
        info_payload,
        source_ego_key=source_ego_key,
        source_wrist_key=source_wrist_key,
    )
    camera_keys = tuple(binding.source_key for binding in cameras)
    return {
        "dataset_content_fingerprint": dataset_content_fingerprint(
            dataset_dir,
            camera_keys=camera_keys,
        ),
        "checkpoint_fingerprint": checkpoint_sha,
        "processor_fingerprint": processor_fingerprint(processor_path),
        "encoder_artifact_fingerprint": f"sha256:{file_sha256(encoder_path)}",
        "prefix_cache_fingerprint": cache_contract.fingerprint,
        "vlm_deployment_content_fingerprint": vlm_content_fingerprint(vlm_model_path),
        "vlm_fingerprint_scope": "deployment_only_not_representation_training_lineage",
        "model_path": str(model_path),
        "processor_path": str(processor_path),
        "vlm_model_path": str(vlm_model_path),
        "prefix_cache_manifest_path": cache_contract.manifest_path,
        "token_contract_fingerprint": token_contract_fingerprint(
            token_scope=token_scope,
            token_sampling=token_sampling,
            max_vl_tokens=max_vl_tokens,
        ),
        "camera_contract_fingerprint": camera_contract_fingerprint(cameras),
        "camera_bindings": [dataclasses.asdict(item) for item in cameras],
    }


def create_real_offline_provider(
    *,
    dataset_dir: str | Path,
    dataset_fingerprint: str,
    info_payload: Mapping[str, Any],
    model_path: str | Path,
    processor_path: str | Path,
    vlm_model_path: str | Path,
    encoder_artifact: str | Path,
    prefix_cache_manifest: str | Path,
    expected_checkpoint_fingerprint: str,
    expected_dataset_content_fingerprint: str,
    expected_processor_fingerprint: str,
    expected_encoder_artifact_fingerprint: str,
    expected_prefix_cache_fingerprint: str,
    expected_vlm_content_fingerprint: str,
    expected_token_contract_fingerprint: str,
    expected_camera_contract_fingerprint: str,
    cache_dir: str | Path,
    token_scope: str,
    token_sampling: str,
    max_vl_tokens: int,
    denoise_steps: int,
    base_seed: int,
    device: str,
    embodiment_tag: str = "new_embodiment",
    instruction_override: str | None = None,
    source_ego_key: str = "observation.images.ego_view",
    source_wrist_key: str = "observation.images.wrist_view",
    groot_repo_path: str | Path | None = None,
) -> OfflineV3MachineAFeatureProvider:
    """Strictly load 400k plus an encoder-only artifact and create the provider."""

    for name, value in (
        ("dataset_fingerprint", dataset_fingerprint),
        ("expected_dataset_content_fingerprint", expected_dataset_content_fingerprint),
        ("expected_checkpoint_fingerprint", expected_checkpoint_fingerprint),
        ("expected_processor_fingerprint", expected_processor_fingerprint),
        ("expected_encoder_artifact_fingerprint", expected_encoder_artifact_fingerprint),
        ("expected_prefix_cache_fingerprint", expected_prefix_cache_fingerprint),
        ("expected_vlm_content_fingerprint", expected_vlm_content_fingerprint),
        ("expected_token_contract_fingerprint", expected_token_contract_fingerprint),
        ("expected_camera_contract_fingerprint", expected_camera_contract_fingerprint),
    ):
        _require_sha256(name, value)
    inspected = inspect_deployment_contract(
        dataset_dir=dataset_dir,
        model_path=model_path,
        processor_path=processor_path,
        vlm_model_path=vlm_model_path,
        encoder_artifact=encoder_artifact,
        prefix_cache_manifest=prefix_cache_manifest,
        info_payload=info_payload,
        source_ego_key=source_ego_key,
        source_wrist_key=source_wrist_key,
        token_scope=token_scope,
        token_sampling=token_sampling,
        max_vl_tokens=max_vl_tokens,
    )
    expected_pairs = {
        "dataset_content_fingerprint": expected_dataset_content_fingerprint,
        "checkpoint_fingerprint": expected_checkpoint_fingerprint,
        "processor_fingerprint": expected_processor_fingerprint,
        "encoder_artifact_fingerprint": expected_encoder_artifact_fingerprint,
        "prefix_cache_fingerprint": expected_prefix_cache_fingerprint,
        "vlm_deployment_content_fingerprint": expected_vlm_content_fingerprint,
        "token_contract_fingerprint": expected_token_contract_fingerprint,
        "camera_contract_fingerprint": expected_camera_contract_fingerprint,
    }
    for name, expected in expected_pairs.items():
        if inspected[name] != expected:
            raise ValueError(f"{name} mismatch: expected {expected}, got {inspected[name]}")

    from groot_rlt.groot_repo import ensure_groot_repo

    ensure_groot_repo(groot_repo_path)
    import torch

    from groot_rlt.representation.encoder_artifact import load_encoder_ema_artifact
    from groot_rlt.representation.precompute_rl_tokens_and_vla_actions import (
        PrecomputeCheckpointRokaePolicy,
    )
    from groot_rlt.representation.train_vl_embedding_autoencoder import pack_vl_tokens

    loaded_encoder = load_encoder_ema_artifact(
        encoder_artifact,
        expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
        expected_cache_fingerprint=expected_prefix_cache_fingerprint,
        device=device,
    )
    cameras = tuple(CameraBinding(**item) for item in inspected["camera_bindings"])
    language_key = "annotation.human.action.task_description"
    contract = OfflineFeatureContract(
        dataset_fingerprint=dataset_fingerprint,
        dataset_content_fingerprint=expected_dataset_content_fingerprint,
        checkpoint_fingerprint=expected_checkpoint_fingerprint,
        encoder_artifact_fingerprint=expected_encoder_artifact_fingerprint,
        processor_fingerprint=expected_processor_fingerprint,
        prefix_cache_fingerprint=expected_prefix_cache_fingerprint,
        vlm_deployment_content_fingerprint=expected_vlm_content_fingerprint,
        token_contract_fingerprint=expected_token_contract_fingerprint,
        camera_contract_fingerprint=expected_camera_contract_fingerprint,
        model_path=inspected["model_path"],
        processor_path=inspected["processor_path"],
        vlm_model_path=inspected["vlm_model_path"],
        prefix_cache_manifest_path=inspected["prefix_cache_manifest_path"],
        token_scope=token_scope,
        token_sampling=token_sampling,
        max_vl_tokens=max_vl_tokens,
        denoise_steps=denoise_steps,
        base_seed=base_seed,
        z_dim=int(loaded_encoder.encoder.config.rl_token_dim),
        camera_bindings=cameras,
        language_key=language_key,
        embodiment_tag=embodiment_tag,
    )
    policy = PrecomputeCheckpointRokaePolicy(
        model_path=Path(model_path).expanduser().resolve(),
        processor_path=Path(processor_path).expanduser().resolve(),
        device=device,
        strict=True,
        vlm_model_path=Path(vlm_model_path).expanduser().resolve(),
        embodiment_tag=embodiment_tag,
    )
    backend = GrootOfflineMachineABackend(
        policy=policy,
        encoder=loaded_encoder.encoder,
        pack_vl_tokens=pack_vl_tokens,
        torch_module=torch,
        device=device,
        contract=contract,
    )
    cache = AtomicFeatureCache(cache_dir, contract)
    return OfflineV3MachineAFeatureProvider(
        backend,
        contract,
        cache,
        instruction_override=instruction_override,
    )


def write_offline_bridge_bundle(
    bundle: ReplayBridgeBundle,
    output_path: str | Path,
) -> dict[str, Any]:
    """Atomically persist the actor/critic bridge payload without overwriting."""

    path = write_replay_bundle(bundle, output_path, overwrite=False)
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "sha256": f"sha256:{file_sha256(path)}",
        "size_bytes": path.stat().st_size,
    }


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cache-dir", required=False)
    parser.add_argument(
        "--bridge-output",
        required=False,
        help="Required outside --inspect-contract; existing files are never overwritten.",
    )
    parser.add_argument("--groot-repo-path", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--vlm-model-path", required=True)
    parser.add_argument("--encoder-artifact", required=True)
    parser.add_argument("--prefix-cache-manifest", required=True)
    parser.add_argument("--expected-dataset-fingerprint", required=False)
    parser.add_argument("--expected-dataset-content-fingerprint", required=False)
    parser.add_argument("--expected-checkpoint-fingerprint", required=False)
    parser.add_argument("--expected-processor-fingerprint", required=False)
    parser.add_argument("--expected-encoder-artifact-fingerprint", required=False)
    parser.add_argument("--expected-prefix-cache-fingerprint", required=False)
    parser.add_argument("--expected-vlm-content-fingerprint", required=False)
    parser.add_argument("--expected-token-contract-fingerprint", required=False)
    parser.add_argument("--expected-camera-contract-fingerprint", required=False)
    parser.add_argument("--token-scope", choices=("all", "image", "non_image"), default="image")
    parser.add_argument(
        "--token-sampling",
        choices=("head", "tail", "uniform"),
        default="uniform",
    )
    parser.add_argument("--max-vl-tokens", type=int, default=192)
    parser.add_argument("--denoise-steps", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--loader",
        choices=("auto", "official", "local"),
        default="auto",
        help="Use official LeRobot, strict local Parquet+PyAV, or official-first auto mode.",
    )
    parser.add_argument("--source-ego-key", default="observation.images.ego_view")
    parser.add_argument("--source-wrist-key", default="observation.images.wrist_view")
    parser.add_argument(
        "--inspect-contract",
        action="store_true",
        help=(
            "Validate dataset rows and print the first-approval dataset plus deployment "
            "fingerprints without loading GR00T models or decoding videos."
        ),
    )
    return parser


def _required_arg(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if not value:
        option = "--" + name.replace("_", "-")
        raise ValueError(f"{option} is required unless --inspect-contract is used")
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = make_arg_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    info = _read_json(dataset_dir / "meta" / "info.json")
    recap = _read_json(dataset_dir / "meta" / "teleop_stack_recap.json")
    inspected = inspect_deployment_contract(
        dataset_dir=dataset_dir,
        model_path=args.model_path,
        processor_path=args.processor_path,
        vlm_model_path=args.vlm_model_path,
        encoder_artifact=args.encoder_artifact,
        prefix_cache_manifest=args.prefix_cache_manifest,
        info_payload=info,
        source_ego_key=args.source_ego_key,
        source_wrist_key=args.source_wrist_key,
        token_scope=args.token_scope,
        token_sampling=args.token_sampling,
        max_vl_tokens=args.max_vl_tokens,
    )
    if args.inspect_contract:
        cameras = tuple(item["source_key"] for item in inspected["camera_bindings"])
        loader = LocalV3ParquetPyAVLoader(
            dataset_dir,
            info_payload=info,
            camera_keys=cameras,
            load_videos=False,
        )
        try:
            inspected.update(
                inspect_lerobot_v3_replay_source(
                    loader,
                    info_payload=info,
                    recap_payload=recap,
                )
            )
        finally:
            loader.close()
        print(json.dumps(inspected, indent=2, sort_keys=True))
        return 0

    bridge_output = Path(_required_arg(args, "bridge_output")).expanduser().resolve()
    if bridge_output.exists():
        raise FileExistsError(f"replay bridge output already exists: {bridge_output}")
    expected_dataset_fingerprint = _required_arg(args, "expected_dataset_fingerprint")
    provider = create_real_offline_provider(
        dataset_dir=dataset_dir,
        dataset_fingerprint=expected_dataset_fingerprint,
        info_payload=info,
        model_path=args.model_path,
        processor_path=args.processor_path,
        vlm_model_path=_required_arg(args, "vlm_model_path"),
        encoder_artifact=args.encoder_artifact,
        prefix_cache_manifest=args.prefix_cache_manifest,
        expected_dataset_content_fingerprint=_required_arg(
            args, "expected_dataset_content_fingerprint"
        ),
        expected_checkpoint_fingerprint=_required_arg(args, "expected_checkpoint_fingerprint"),
        expected_processor_fingerprint=_required_arg(args, "expected_processor_fingerprint"),
        expected_encoder_artifact_fingerprint=_required_arg(
            args, "expected_encoder_artifact_fingerprint"
        ),
        expected_prefix_cache_fingerprint=_required_arg(args, "expected_prefix_cache_fingerprint"),
        expected_vlm_content_fingerprint=_required_arg(
            args, "expected_vlm_content_fingerprint"
        ),
        expected_token_contract_fingerprint=_required_arg(
            args, "expected_token_contract_fingerprint"
        ),
        expected_camera_contract_fingerprint=_required_arg(
            args, "expected_camera_contract_fingerprint"
        ),
        cache_dir=_required_arg(args, "cache_dir"),
        token_scope=args.token_scope,
        token_sampling=args.token_sampling,
        max_vl_tokens=args.max_vl_tokens,
        denoise_steps=args.denoise_steps,
        base_seed=args.base_seed,
        device=args.device,
        embodiment_tag=args.embodiment_tag,
        instruction_override=args.instruction,
        source_ego_key=args.source_ego_key,
        source_wrist_key=args.source_wrist_key,
        groot_repo_path=args.groot_repo_path,
    )
    repo_id = recap.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("teleop_stack_recap.repo_id must be non-empty")
    loader = open_v3_dataset_loader(
        repo_id=repo_id,
        root=dataset_dir,
        info_payload=info,
        camera_keys=(args.source_ego_key, args.source_wrist_key),
        loader_mode=args.loader,
        video_backend=args.video_backend,
    )
    bundle = build_lerobot_v3_replay_bundle(
        loader,
        info_payload=info,
        recap_payload=recap,
        feature_provider=provider,
        feature_contract_fingerprint=provider.feature_contract_fingerprint,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        collection_phase="warmup",
    )
    bridge_artifact = write_offline_bridge_bundle(bundle, bridge_output)
    print(
        json.dumps(
            {
                "bridge_manifest": bundle.manifest(),
                "bridge_output": bridge_artifact,
                "feature_cache": provider.cache.summary(),
                "cache_hits": provider.cache_hits,
                "cache_misses": provider.cache_misses,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
