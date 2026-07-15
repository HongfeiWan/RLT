from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import hashlib
import http
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import logging
import os
import pickle
import re
import threading
import time
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
from websockets import exceptions as websocket_exceptions
import websockets.sync.client

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.config import ActorServiceConfig
from rlt_online_rl.config import EnvDriverConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import assert_action_contract_matches
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import PyTree
from rlt_online_rl.replay import DEFAULT_COLLECTION_PHASE
from rlt_online_rl.replay import RawEpisodeChunk
from rlt_online_rl.replay import RawEpisodeStep
from rlt_online_rl.replay import RawEpisodeTrace
from rlt_online_rl.replay import ReplayClient
from rlt_online_rl.replay import RLTTransition
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.replay import raw_episode_path_for
from rlt_online_rl.replay import save_raw_episode
from rlt_online_rl.runtime_logging import append_jsonl

logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_STRICT_RLT_ARCHITECTURE = "openpi_rlt_strict_cross_attention_v1"
_STRICT_ENCODER_ARCHITECTURE = f"{_STRICT_RLT_ARCHITECTURE}.encoder_only"


def _validate_groot_machine_a_metadata(metadata: Mapping[str, Any]) -> None:
    """Fail closed when a GR00T feature server cannot prove its prefix lineage."""

    contract = metadata.get("rl_token_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("GR00T Machine A metadata is missing rl_token_contract")
    required = {
        "source_kind",
        "artifact_kind",
        "architecture",
        "source_architecture",
        "artifact_path",
        "artifact_manifest_path",
        "artifact_fingerprint",
        "artifact_file_sha256",
        "encoder_tensor_sha256",
        "representation_checkpoint_file_sha256",
        "checkpoint_fingerprint",
        "cache_fingerprint",
        "representation_checkpoint_schema_version",
        "prefix_cache_schema_version",
        "feature_tap",
        "processor_mode",
        "prefix_cache_manifest_path",
        "token_scope",
        "token_sampling",
        "max_vl_tokens",
        "input_dim",
        "z_dim",
        "video_modality_keys",
        "model_path",
        "processor_path",
        "vlm_model_path",
        "vlm_deployment_content_fingerprint",
        "vlm_fingerprint_scope",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"GR00T rl_token_contract is missing fields: {missing}")
    for name in (
        "artifact_fingerprint",
        "artifact_file_sha256",
        "encoder_tensor_sha256",
        "representation_checkpoint_file_sha256",
        "checkpoint_fingerprint",
        "cache_fingerprint",
        "vlm_deployment_content_fingerprint",
    ):
        if not isinstance(contract[name], str) or _SHA256_PATTERN.fullmatch(contract[name]) is None:
            raise ValueError(f"GR00T rl_token_contract.{name} is not a SHA-256 fingerprint")
    if contract["source_architecture"] != _STRICT_RLT_ARCHITECTURE:
        raise ValueError("GR00T Machine A did not load the strict RLT architecture")
    source_kind = contract["source_kind"]
    if source_kind == "encoder_ema_artifact":
        expected_architecture = _STRICT_ENCODER_ARCHITECTURE
        expected_kind = "groot_rlt.encoder_ema"
        if not isinstance(contract["artifact_manifest_path"], str) or not contract["artifact_manifest_path"]:
            raise ValueError("Encoder-only artifact handshake is missing its manifest path")
    elif source_kind == "legacy_full_checkpoint":
        expected_architecture = _STRICT_RLT_ARCHITECTURE
        expected_kind = "groot_rlt.legacy_full_training_checkpoint"
        if contract["artifact_manifest_path"] is not None:
            raise ValueError("Legacy full checkpoint must not advertise an artifact manifest")
    else:
        raise ValueError(f"Unsupported GR00T RL-token source_kind={source_kind!r}")
    if contract["architecture"] != expected_architecture:
        raise ValueError(
            f"GR00T RL-token architecture={contract['architecture']!r}, expected={expected_architecture!r}"
        )
    if contract["artifact_kind"] != expected_kind:
        raise ValueError(f"GR00T RL-token artifact_kind={contract['artifact_kind']!r} is invalid")
    for name in ("artifact_path", "prefix_cache_manifest_path"):
        if not isinstance(contract[name], str) or not contract[name]:
            raise ValueError(f"GR00T rl_token_contract.{name} must be a non-empty path")
    for name in ("model_path", "processor_path", "vlm_model_path"):
        if not isinstance(contract[name], str) or not contract[name].startswith("/"):
            raise ValueError(f"GR00T rl_token_contract.{name} must be an absolute path")
    if contract["vlm_fingerprint_scope"] != "deployment_only_not_representation_training_lineage":
        raise ValueError("GR00T VLM fingerprint scope is missing or misleading")
    if contract["representation_checkpoint_schema_version"] != 2:
        raise ValueError("GR00T representation checkpoint must use lineage schema 2")
    if contract["prefix_cache_schema_version"] != 2:
        raise ValueError("GR00T prefix cache must use schema 2")
    if contract["feature_tap"] != "raw_backbone_pre_action_head":
        raise ValueError("GR00T prefix feature tap is not serving-equivalent")
    if contract["processor_mode"] != "eval":
        raise ValueError("GR00T prefix processor mode must be eval")
    if contract["token_scope"] not in {"all", "image", "non_image"}:
        raise ValueError("GR00T token scope is invalid")
    if contract["token_sampling"] not in {"head", "tail", "uniform"}:
        raise ValueError("GR00T token sampling must be deterministic")
    for name in ("max_vl_tokens", "input_dim", "z_dim"):
        if type(contract[name]) is not int or contract[name] < 1:
            raise ValueError(f"GR00T rl_token_contract.{name} must be a positive integer")
    cameras = contract["video_modality_keys"]
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, str) and camera for camera in cameras)
        or len(set(cameras)) != len(cameras)
    ):
        raise ValueError("GR00T rl_token_contract.video_modality_keys is invalid")
    for top_level, nested in (
        ("z_dim", "z_dim"),
        ("token_scope", "token_scope"),
        ("token_sampling", "token_sampling"),
        ("max_vl_tokens", "max_vl_tokens"),
    ):
        if metadata.get(top_level) != contract[nested]:
            raise ValueError(
                f"GR00T handshake {top_level}={metadata.get(top_level)!r} conflicts with "
                f"rl_token_contract.{nested}={contract[nested]!r}"
            )


def _validate_machine_a_metadata(
    metadata: Mapping[str, Any],
    *,
    allow_unpinned: bool,
    expected_backend: str | None,
    expected_checkpoint_fingerprint: str | None,
    expected_cache_fingerprint: str | None,
    expected_encoder_artifact_sha256: str | None,
    expected_vlm_content_fingerprint: str | None,
) -> None:
    """Validate server metadata against an independently configured identity."""

    expected = {
        "expected_backend": expected_backend,
        "expected_checkpoint_fingerprint": expected_checkpoint_fingerprint,
        "expected_cache_fingerprint": expected_cache_fingerprint,
        "expected_encoder_artifact_sha256": expected_encoder_artifact_sha256,
        "expected_vlm_content_fingerprint": expected_vlm_content_fingerprint,
    }
    if not allow_unpinned:
        missing = sorted(name for name, value in expected.items() if not value)
        if missing:
            raise ValueError(f"Pinned Machine A metadata requires configured values: {missing}")
    for name, value in expected.items():
        if name == "expected_backend" or value is None:
            continue
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Configured {name} is not a SHA-256 fingerprint")

    backend = metadata.get("backend")
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(
            f"Machine A backend={backend!r} does not match configured {expected_backend!r}"
        )
    must_validate_groot = backend == "groot-n1.7" or expected_backend == "groot-n1.7"
    if not must_validate_groot:
        if allow_unpinned:
            return
        raise ValueError(f"Pinned Machine A backend {backend!r} has no strict validator")

    _validate_groot_machine_a_metadata(metadata)
    contract = metadata["rl_token_contract"]
    comparisons = {
        "checkpoint_fingerprint": expected_checkpoint_fingerprint,
        "cache_fingerprint": expected_cache_fingerprint,
        "artifact_file_sha256": expected_encoder_artifact_sha256,
        "vlm_deployment_content_fingerprint": expected_vlm_content_fingerprint,
    }
    for name, expected_value in comparisons.items():
        if expected_value is not None and contract[name] != expected_value:
            raise ValueError(
                f"Machine A rl_token_contract.{name}={contract[name]!r} does not match "
                f"configured {expected_value!r}"
            )


def _healthz_url_from_ws_url(ws_url: str) -> str:
    parsed = urllib_parse.urlsplit(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urllib_parse.urlunsplit((scheme, parsed.netloc, "/healthz", "", ""))


@dataclasses.dataclass(slots=True)
class ActorRequest:
    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    request_id: str
    episode_id: int
    step_id: int
    deterministic: bool = False
    timestamp: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "z_rl": np.asarray(self.z_rl, dtype=np.float32),
            "proprio": np.asarray(self.proprio, dtype=np.float32),
            "ref_chunk": np.asarray(self.ref_chunk, dtype=np.float32),
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "deterministic": self.deterministic,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass(slots=True)
class ActorResponse:
    refined_chunk: np.ndarray
    actor_param_version: int
    request_id: str
    timestamp: float | None = None
    source: int = int(TransitionSource.RL)


@dataclasses.dataclass(slots=True)
class RefinementResult:
    refined_chunk: np.ndarray
    source: int
    actor_param_version: int
    used_fallback: bool
    error: str | None = None


@dataclasses.dataclass(slots=True)
class StepTraceRecord:
    observation: dict[str, Any]
    action: np.ndarray
    ref_action: np.ndarray
    reward: float
    next_observation: dict[str, Any]
    source: int
    collection_phase: str
    human_controlled: bool
    done: bool
    success: int
    episode_id: int
    env_step_id: int
    actor_param_version: int = -1


class FeatureProvider(Protocol):
    def get_features(self, _observation: dict[str, Any]) -> dict[str, Any]: ...


@dataclasses.dataclass(slots=True)
class ChunkFeatures:
    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    source_ref_chunk: np.ndarray | None = None


@dataclasses.dataclass(slots=True)
class ReplaySegment:
    raw_indices: list[int]


@dataclasses.dataclass(slots=True)
class ReplayWindow:
    segment_id: int
    start_offset: int


@dataclasses.dataclass(slots=True)
class PolicyPlan:
    action_chunk: np.ndarray
    ref_chunk: np.ndarray
    source: int
    start_features: ChunkFeatures
    actor_param_version: int = -1


def _coerce_feature_vector(name: str, value: Any, expected_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1 or [1, D], got shape {array.shape}.")
    if array.shape[0] != expected_dim:
        raise ValueError(f"{name} expected dim {expected_dim}, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _coerce_ref_chunk(ref_chunk: Any, *, min_chunk_len: int, action_dim: int) -> np.ndarray:
    array = np.asarray(ref_chunk, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"ref_chunk must be rank-2 or [1, T, A], got shape {array.shape}.")
    if array.shape[0] < min_chunk_len or array.shape[1] != action_dim:
        raise ValueError(f"ref_chunk must have shape [>={min_chunk_len}, {action_dim}], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("ref_chunk contains non-finite values.")
    return array


def _coerce_action_chunk(name: str, value: Any, *, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _coerce_action_vector(name: str, value: Any, *, action_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (action_dim,):
        raise ValueError(f"{name} must have shape ({action_dim},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _semantic_layout_hash(channel_names: list[str], rotation_convention: str | None) -> str:
    material = {
        "channel_names": list(channel_names),
        "rotation_convention": rotation_convention,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _project_machine_a_reference(
    payload: dict[str, Any],
    rl_config: RLTOnlineRLConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    source_dim = rl_config.reference_action_dim
    indices = rl_config.reference_action_indices
    source_hash = rl_config.reference_action_layout_hash
    projected_hash = rl_config.action_layout_hash

    projection_configured = any(value is not None for value in (source_dim, indices, source_hash))
    if not projection_configured:
        source = _coerce_ref_chunk(
            payload["ref_chunk"],
            min_chunk_len=rl_config.chunk_len,
            action_dim=rl_config.action_dim,
        )[: rl_config.chunk_len]
        if projected_hash is not None and payload.get("action_layout_hash") != projected_hash:
            raise ValueError(
                f"Machine A action_layout_hash={payload.get('action_layout_hash')!r} does not "
                f"match configured {projected_hash!r}"
            )
        return source.copy(), None

    if source_dim is None or indices is None or source_hash is None or projected_hash is None:
        raise ValueError(
            "reference projection requires reference_action_dim, reference_action_indices, "
            "reference_action_layout_hash, and action_layout_hash"
        )
    if len(indices) != rl_config.action_dim:
        raise ValueError(f"reference_action_indices has {len(indices)} entries, expected {rl_config.action_dim}")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= source_dim for index in indices):
        raise ValueError(f"reference_action_indices are invalid for source dim {source_dim}: {indices}")

    source = _coerce_ref_chunk(
        payload["ref_chunk"],
        min_chunk_len=rl_config.chunk_len,
        action_dim=source_dim,
    )[: rl_config.chunk_len]
    if payload.get("action_layout_hash") != source_hash:
        raise ValueError(
            f"Machine A source action_layout_hash={payload.get('action_layout_hash')!r} does not "
            f"match configured reference_action_layout_hash={source_hash!r}"
        )
    source_layout = payload.get("action_layout")
    if not isinstance(source_layout, (list, tuple)) or len(source_layout) != source_dim:
        raise ValueError(f"Machine A action_layout must contain {source_dim} source names for projection")
    if rl_config.rot6d_convention is not None and payload.get("rot6d_convention") != rl_config.rot6d_convention:
        raise ValueError(
            f"Machine A rot6d_convention={payload.get('rot6d_convention')!r} does not match "
            f"configured {rl_config.rot6d_convention!r}"
        )
    projected_layout = [str(source_layout[index]) for index in indices]
    actual_projected_hash = _semantic_layout_hash(projected_layout, rl_config.rot6d_convention)
    if actual_projected_hash != projected_hash:
        raise ValueError(
            f"projected Machine A action layout hash {actual_projected_hash!r} does not match "
            f"configured action_layout_hash={projected_hash!r}"
        )
    projected = source[:, np.asarray(indices, dtype=np.int64)]
    return projected.astype(np.float32, copy=True), source.astype(np.float32, copy=True)


def _proprio_from_observation(observation: dict[str, Any], rl_config: RLTOnlineRLConfig) -> np.ndarray:
    if "state" not in observation:
        raise ValueError("observation missing required key 'state'.")
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    expected_exact = rl_config.proprio_layout_hash is not None
    invalid_dim = state.ndim != 1 or (
        state.shape[0] != rl_config.proprio_dim if expected_exact else state.shape[0] < rl_config.proprio_dim
    )
    if invalid_dim:
        expectation = "exactly" if expected_exact else "at least"
        raise ValueError(
            f"observation state must be rank-1 with {expectation} dim {rl_config.proprio_dim}, got shape {state.shape}."
        )
    return state[: rl_config.proprio_dim].astype(np.float32, copy=False)


def normalize_feature_payload(
    payload: dict[str, Any],
    rl_config: RLTOnlineRLConfig,
    *,
    observation: dict[str, Any],
) -> dict[str, Any]:
    required = {"z_rl", "ref_chunk"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Machine A feature payload missing required keys: {missing}")

    normalized = dict(payload)
    normalized["z_rl"] = _coerce_feature_vector("z_rl", payload["z_rl"], rl_config.z_dim)
    observation_state = observation.get("state")
    pinned_proprio = rl_config.proprio_layout_hash is not None
    if pinned_proprio:
        if "proprio" not in payload:
            raise ValueError(
                "Pinned Machine A proprio contract requires an explicit payload proprio vector; "
                "the raw observation state cannot be sliced implicitly."
            )
        normalized["proprio"] = _coerce_feature_vector("proprio", payload["proprio"], rl_config.proprio_dim)
    elif isinstance(observation_state, Mapping):
        if "proprio" not in payload:
            raise ValueError("nested GR00T observation state requires Machine A to return a flat proprio vector.")
        normalized["proprio"] = _coerce_feature_vector("proprio", payload["proprio"], rl_config.proprio_dim)
    else:
        normalized["proprio"] = _proprio_from_observation(observation, rl_config)
    projected_ref, source_ref = _project_machine_a_reference(payload, rl_config)
    normalized["ref_chunk"] = projected_ref
    if source_ref is not None:
        normalized["source_ref_chunk"] = source_ref
    for payload_key, expected in (("proprio_layout_hash", rl_config.proprio_layout_hash),):
        if expected is not None and payload.get(payload_key) != expected:
            raise ValueError(
                f"Machine A {payload_key}={payload.get(payload_key)!r} does not match configured {expected!r}"
            )
    if pinned_proprio:
        layout = payload.get("proprio_layout")
        if (
            not isinstance(layout, (list, tuple))
            or len(layout) != rl_config.proprio_dim
            or not all(isinstance(name, str) and name for name in layout)
        ):
            raise ValueError(
                f"Machine A proprio_layout must contain {rl_config.proprio_dim} non-empty names"
            )
        actual_layout_hash = _semantic_layout_hash(
            list(layout),
            rl_config.rot6d_convention,
        )
        if actual_layout_hash != rl_config.proprio_layout_hash:
            raise ValueError(
                f"Machine A proprio layout hash {actual_layout_hash!r} does not match "
                f"configured proprio_layout_hash={rl_config.proprio_layout_hash!r}"
            )
    return normalized


def _chunk_features_from_payload(payload: dict[str, Any], rl_config: RLTOnlineRLConfig) -> ChunkFeatures:
    return ChunkFeatures(
        z_rl=np.asarray(payload["z_rl"], dtype=np.float32),
        proprio=np.asarray(payload["proprio"], dtype=np.float32),
        ref_chunk=np.asarray(payload["ref_chunk"], dtype=np.float32),
        source_ref_chunk=None
        if "source_ref_chunk" not in payload
        else np.asarray(payload["source_ref_chunk"], dtype=np.float32),
    )


def _normalize_cached_feature_payload(payload: dict[str, Any], rl_config: RLTOnlineRLConfig) -> dict[str, np.ndarray]:
    normalized = {
        "z_rl": _coerce_feature_vector("z_rl", payload["z_rl"], rl_config.z_dim),
        "proprio": _coerce_feature_vector("proprio", payload["proprio"], rl_config.proprio_dim),
        "ref_chunk": _coerce_ref_chunk(
            payload["ref_chunk"],
            min_chunk_len=rl_config.chunk_len,
            action_dim=rl_config.action_dim,
        )[: rl_config.chunk_len],
    }
    if "source_ref_chunk" in payload:
        source_dim = rl_config.reference_action_dim or rl_config.action_dim
        normalized["source_ref_chunk"] = _coerce_ref_chunk(
            payload["source_ref_chunk"],
            min_chunk_len=rl_config.chunk_len,
            action_dim=source_dim,
        )[: rl_config.chunk_len]
    return normalized


class RLTPolicyInferenceWrapper:
    """Pure JAX inference wrapper for B1.

    It only runs the actor mean. It does not own checkpointing, polling or RPC.
    """

    def __init__(self, rl_config: RLTOnlineRLConfig):
        self._rl_config = rl_config
        self._actor = ChunkActor(
            z_dim=rl_config.z_dim,
            proprio_dim=rl_config.proprio_dim,
            chunk_len=rl_config.chunk_len,
            action_dim=rl_config.action_dim,
            hidden_dim=rl_config.actor_hidden_dim,
            num_layers=rl_config.actor_num_layers,
            fixed_std=rl_config.fixed_std,
        )
        self._compiled_mean = jax.jit(self._forward_mean)
        self._compiled_sample = jax.jit(self._forward_sample)

    def infer(
        self,
        actor_params: PyTree,
        z_rl: np.ndarray,
        proprio: np.ndarray,
        ref_chunk: np.ndarray,
        *,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        z_rl = np.asarray(z_rl, dtype=np.float32)[None, ...]
        proprio = np.asarray(proprio, dtype=np.float32)[None, ...]
        ref_chunk = np.asarray(ref_chunk, dtype=np.float32)[None, ...]
        if deterministic:
            output = self._compiled_mean(actor_params, jnp.asarray(z_rl), jnp.asarray(proprio), jnp.asarray(ref_chunk))
        else:
            if rng is None:
                raise ValueError("rng is required for stochastic actor inference.")
            output = self._compiled_sample(
                actor_params,
                rng,
                jnp.asarray(z_rl),
                jnp.asarray(proprio),
                jnp.asarray(ref_chunk),
            )
        return np.asarray(jax.device_get(output[0]))

    def _forward_mean(
        self, actor_params: PyTree, z_rl: jax.Array, proprio: jax.Array, ref_chunk: jax.Array
    ) -> jax.Array:
        return self._actor.actor_mean(actor_params, z_rl, proprio, ref_chunk)

    def _forward_sample(
        self,
        actor_params: PyTree,
        rng: jax.Array,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> jax.Array:
        return self._actor.sample_action(actor_params, rng, z_rl, proprio, ref_chunk, deterministic=False)


class ActorService:
    """B1 process core object and local inference HTTP server."""

    def __init__(self, rl_config: RLTOnlineRLConfig, service_config: ActorServiceConfig):
        self._rl_config = rl_config
        self._service_config = service_config
        self._snapshot_path = service_config.snapshot_path
        self._action_adapter = ActionRepresentationAdapter.from_config(rl_config)
        self._wrapper = RLTPolicyInferenceWrapper(rl_config)
        self._actor_params: PyTree | None = None
        self._actor_version = -1
        self._rng = jax.random.PRNGKey(0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._logged_missing_params = False
        self._packer = msgpack_numpy.Packer()
        # Load an existing snapshot before accepting requests.  Falling back to
        # the base chunk while the background poller is still compiling the
        # initial actor is unsafe and makes startup timing machine-dependent.
        self._try_reload_snapshot()
        self._start_param_poller()

    def infer(self, request: ActorRequest) -> ActorResponse:
        z_rl = _coerce_feature_vector("actor request z_rl", request.z_rl, self._rl_config.z_dim)
        proprio = _coerce_feature_vector(
            "actor request proprio",
            request.proprio,
            self._rl_config.proprio_dim,
        )
        ref_chunk = _coerce_action_chunk(
            "actor request ref_chunk",
            request.ref_chunk,
            expected_shape=(self._rl_config.chunk_len, self._rl_config.action_dim),
        )
        with self._lock:
            actor_params = self._actor_params
            actor_version = self._actor_version
            infer_rng = None
            if actor_params is not None and not request.deterministic:
                self._rng, infer_rng = jax.random.split(self._rng)
        if actor_params is None:
            if not self._logged_missing_params:
                logger.info("No actor snapshot loaded yet; falling back to ref_chunk.")
                self._logged_missing_params = True
            return ActorResponse(
                refined_chunk=ref_chunk.copy(),
                actor_param_version=actor_version,
                request_id=request.request_id,
                timestamp=time.time(),
                source=int(TransitionSource.BASE),
            )
        model_ref_chunk = ref_chunk
        if self._action_adapter is not None:
            model_ref_chunk = self._action_adapter.normalize_ref_chunk(model_ref_chunk, proprio)
        refined_chunk = self._wrapper.infer(
            actor_params,
            z_rl,
            proprio,
            model_ref_chunk,
            rng=infer_rng,
            deterministic=request.deterministic,
        )
        if self._action_adapter is not None:
            refined_chunk = self._action_adapter.denormalize_to_abs_chunk(refined_chunk, proprio)
        refined_chunk = _coerce_action_chunk(
            "actor response refined_chunk",
            refined_chunk,
            expected_shape=ref_chunk.shape,
        )
        return ActorResponse(
            refined_chunk=refined_chunk,
            actor_param_version=actor_version,
            request_id=request.request_id,
            timestamp=time.time(),
            source=int(TransitionSource.RL),
        )

    def warmup_inference(self, actor_params: PyTree) -> None:
        z_rl = np.zeros((self._rl_config.z_dim,), dtype=np.float32)
        proprio = np.zeros((self._rl_config.proprio_dim,), dtype=np.float32)
        ref_chunk = np.zeros((self._rl_config.chunk_len, self._rl_config.action_dim), dtype=np.float32)
        if self._action_adapter is not None:
            ref_chunk = self._action_adapter.normalize_ref_chunk(ref_chunk, proprio)
        self._wrapper.infer(actor_params, z_rl, proprio, ref_chunk, deterministic=True)

    def serve_forever(self, *, stop_event: threading.Event | None = None) -> None:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self.send_response(http.HTTPStatus.OK)
                    self.end_headers()
                    self.wfile.write(b"OK\n")
                    return
                if self.path == "/version":
                    payload = service._packer.pack({"actor_param_version": service.actor_param_version})
                    self.send_response(http.HTTPStatus.OK)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_error(http.HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/infer":
                    self.send_error(http.HTTPStatus.NOT_FOUND)
                    return
                content_len = int(self.headers.get("Content-Length", "0"))
                payload = msgpack_numpy.unpackb(self.rfile.read(content_len))
                response = service.infer(
                    ActorRequest(
                        z_rl=np.asarray(payload["z_rl"], dtype=np.float32),
                        proprio=np.asarray(payload["proprio"], dtype=np.float32),
                        ref_chunk=np.asarray(payload["ref_chunk"], dtype=np.float32),
                        request_id=str(payload["request_id"]),
                        episode_id=int(payload["episode_id"]),
                        step_id=int(payload["step_id"]),
                        deterministic=bool(payload.get("deterministic", False)),
                        timestamp=payload.get("timestamp"),
                    )
                )
                body = service._packer.pack(
                    {
                        "refined_chunk": response.refined_chunk,
                        "actor_param_version": response.actor_param_version,
                        "request_id": response.request_id,
                        "timestamp": response.timestamp,
                        "source": response.source,
                    }
                )
                self.send_response(http.HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("Actor client disconnected before inference response was written.")

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = ThreadingHTTPServer((self._service_config.bind_host, self._service_config.port), Handler)
        server.timeout = 0.5
        logger.debug(
            "ActorService listening on http://%s:%s snapshot_path=%s",
            self._service_config.bind_host,
            self._service_config.port,
            self._snapshot_path,
        )
        try:
            if stop_event is None:
                server.serve_forever()
            else:
                while not stop_event.is_set():
                    server.handle_request()
        finally:
            self._stop_event.set()
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=1.0)
            server.server_close()
            logger.info("ActorService stopped.")

    @property
    def actor_param_version(self) -> int:
        with self._lock:
            return self._actor_version

    def _start_param_poller(self) -> None:
        self._poll_thread = threading.Thread(target=self._poll_snapshot_loop, daemon=True)
        self._poll_thread.start()

    def _poll_snapshot_loop(self) -> None:
        while not self._stop_event.is_set():
            self._try_reload_snapshot()
            self._stop_event.wait(self._service_config.pull_params_interval_sec)

    def _try_reload_snapshot(self) -> None:
        if not os.path.exists(self._snapshot_path):
            return
        try:
            with open(self._snapshot_path, "rb") as f:
                payload = pickle.load(f)
        except (EOFError, FileNotFoundError, pickle.UnpicklingError):
            return
        assert_action_contract_matches(
            payload.get("rl_config"),
            self._rl_config,
            context=f"actor snapshot {self._snapshot_path}",
        )
        version = int(payload["version"])
        with self._lock:
            current_version = self._actor_version
        if version <= current_version:
            return
        actor_params = jax.tree_util.tree_map(jnp.asarray, payload["actor_params"])
        self.warmup_inference(actor_params)
        with self._lock:
            if version <= self._actor_version:
                return
            self._actor_params = actor_params
            self._actor_version = version
            self._logged_missing_params = False
        logger.info("Loaded actor snapshot version=%s", version)


class ActorClient:
    """Local client for the B1 actor_service."""

    def __init__(self, base_url: str, *, timeout_sec: float = 1.0, max_retries: int = 1):
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._packer = msgpack_numpy.Packer()

    def infer(self, request: ActorRequest) -> ActorResponse:
        payload = request.to_payload()
        body = self._packer.pack(payload)
        http_request = urllib_request.Request(
            f"{self._base_url}/infer",
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                with urllib_request.urlopen(http_request, timeout=self._timeout_sec) as response:
                    payload = msgpack_numpy.unpackb(response.read())
                    return ActorResponse(
                        refined_chunk=np.asarray(payload["refined_chunk"], dtype=np.float32),
                        actor_param_version=int(payload["actor_param_version"]),
                        request_id=str(payload["request_id"]),
                        timestamp=payload.get("timestamp"),
                        source=int(payload.get("source", int(TransitionSource.RL))),
                    )
            except (urllib_error.URLError, TimeoutError) as exc:
                last_error = exc
        raise RuntimeError("actor_service inference failed") from last_error

    def get_actor_param_version(self) -> int:
        req = urllib_request.Request(f"{self._base_url}/version", method="GET")
        with urllib_request.urlopen(req, timeout=self._timeout_sec) as response:
            payload = msgpack_numpy.unpackb(response.read())
        return int(payload["actor_param_version"])


def maybe_refine_chunk(
    actor_client: ActorClient,
    *,
    z_rl: np.ndarray,
    proprio: np.ndarray,
    ref_chunk: np.ndarray,
    request_id: str,
    episode_id: int,
    step_id: int,
    deterministic: bool = False,
    on_error_fallback: bool = True,
) -> RefinementResult:
    try:
        response = actor_client.infer(
            ActorRequest(
                z_rl=z_rl,
                proprio=proprio,
                ref_chunk=ref_chunk,
                request_id=request_id,
                episode_id=episode_id,
                step_id=step_id,
                deterministic=deterministic,
                timestamp=time.time(),
            )
        )
        return RefinementResult(
            refined_chunk=response.refined_chunk,
            source=int(response.source),
            actor_param_version=response.actor_param_version,
            used_fallback=False,
        )
    except RuntimeError as exc:
        if not on_error_fallback:
            raise
        return RefinementResult(
            refined_chunk=np.asarray(ref_chunk, dtype=np.float32),
            source=int(TransitionSource.BASE),
            actor_param_version=-1,
            used_fallback=True,
            error=str(exc),
        )


class MachineAFeatureClient:
    """Client stub for the remote feature/reference service on Machine A."""

    def __init__(
        self,
        ws_url: str,
        *,
        connect_timeout_sec: float = 5.0,
        recv_timeout_sec: float = 5.0,
        retry_interval_sec: float = 0.5,
        allow_unpinned_metadata: bool = True,
        expected_backend: str | None = None,
        expected_checkpoint_fingerprint: str | None = None,
        expected_cache_fingerprint: str | None = None,
        expected_encoder_artifact_sha256: str | None = None,
        expected_vlm_content_fingerprint: str | None = None,
    ):
        self._ws_url = ws_url
        self._healthz_url = _healthz_url_from_ws_url(ws_url)
        self._connect_timeout_sec = connect_timeout_sec
        self._recv_timeout_sec = recv_timeout_sec
        self._retry_interval_sec = retry_interval_sec
        self._metadata_expectation = {
            "allow_unpinned": bool(allow_unpinned_metadata),
            "expected_backend": expected_backend,
            "expected_checkpoint_fingerprint": expected_checkpoint_fingerprint,
            "expected_cache_fingerprint": expected_cache_fingerprint,
            "expected_encoder_artifact_sha256": expected_encoder_artifact_sha256,
            "expected_vlm_content_fingerprint": expected_vlm_content_fingerprint,
        }
        self._packer = msgpack_numpy.Packer()
        self._metadata: dict[str, Any] = {}
        self._ws = self._wait_for_server()

    def get_features(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._infer(observation)

    def get_features_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch feature request: send multiple observations in one round-trip.

        The server must support batch inference (metadata["supports_batch"] == True).
        Sends {"batch": [obs1, obs2, ...]} and expects {"batch_results": [r1, r2, ...]}.
        Falls back to sequential get_features() if batch request fails.
        """
        if not observations:
            return []
        if not bool(self._metadata.get("supports_batch", False)):
            return [self.get_features(observation) for observation in observations]
        try:
            response = self._infer({"batch": observations})
            if isinstance(response, dict) and "batch_results" in response:
                results = response["batch_results"]
                if len(results) != len(observations):
                    raise ValueError(f"Batch response length mismatch: sent {len(observations)}, got {len(results)}")
                return results
            raise ValueError(f"Unexpected batch response format: {type(response)}")
        except Exception as exc:
            logger.warning("Batch feature request failed (%s); falling back to sequential.", exc)
            return [self.get_features(obs) for obs in observations]

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def _connect(self):
        return websockets.sync.client.connect(
            self._ws_url,
            compression=None,
            max_size=None,
            open_timeout=self._connect_timeout_sec,
        )

    def _wait_for_server(self):
        logger.debug("Waiting for Machine A feature server at %s", self._ws_url)
        while True:
            ws = None
            try:
                req = urllib_request.Request(self._healthz_url, method="GET")
                with urllib_request.urlopen(req, timeout=self._connect_timeout_sec) as response:
                    if response.status != http.HTTPStatus.OK:
                        raise RuntimeError(f"unexpected healthz status={response.status}")
                ws = self._connect()
                metadata = msgpack_numpy.unpackb(ws.recv(timeout=self._recv_timeout_sec))
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"Machine A metadata must be a mapping, got {type(metadata).__name__}.")
                _validate_machine_a_metadata(metadata, **self._metadata_expectation)
                self._metadata = dict(metadata)
                logger.debug("Connected to Machine A feature server at %s", self._ws_url)
                return ws
            except (
                RuntimeError,
                urllib_error.URLError,
                ConnectionRefusedError,
                OSError,
                TimeoutError,
                websocket_exceptions.ConnectionClosed,
            ) as exc:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                logger.warning("Machine A feature server not ready at %s: %s", self._ws_url, exc)
                time.sleep(self._retry_interval_sec)

    def _reconnect(self) -> None:
        self.close()
        self._ws = self._wait_for_server()

    def _infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = self._packer.pack(observation)
        for attempt in range(2):
            try:
                self._ws.send(payload)
                response = self._ws.recv(timeout=self._recv_timeout_sec)
                break
            except (
                OSError,
                TimeoutError,
                websocket_exceptions.ConnectionClosed,
            ) as exc:
                if attempt == 1:
                    raise RuntimeError(f"Machine A feature request failed: {exc}") from exc
                logger.warning("Machine A websocket request failed; reconnecting to %s: %s", self._ws_url, exc)
                self._reconnect()
        if isinstance(response, str):
            raise RuntimeError(f"Machine A feature service error:\n{response}")
        unpacked = msgpack_numpy.unpackb(response)
        if not isinstance(unpacked, dict):
            raise RuntimeError(f"Machine A feature response must be a mapping, got {type(unpacked).__name__}.")
        return unpacked


class EnvDriver:
    """B4 rollout driver.

    This object only coordinates:
    - Machine A feature/reference RPC
    - B1 actor_service RPC
    - environment execution
    - transition delivery to B3 replay_manager
    """

    def __init__(
        self,
        env: Any,
        feature_provider: FeatureProvider,
        actor_client: ActorClient,
        replay_client: ReplayClient,
        rl_config: RLTOnlineRLConfig,
        env_config: EnvDriverConfig,
        *,
        eval_actor_only: bool = False,
        safe_action_filter: Callable[[np.ndarray], np.ndarray] | None = None,
        human_override_fn: Callable[[dict[str, Any], np.ndarray, np.ndarray], np.ndarray | None] | None = None,
        metrics_path: str | None = None,
    ) -> None:
        self._env = env
        self._feature_provider = feature_provider
        self._actor_client = actor_client
        self._replay_client = replay_client
        self._rl_config = rl_config
        self._env_config = env_config
        self._eval_actor_only = bool(eval_actor_only)
        self._safe_action_filter = safe_action_filter
        self._human_override_fn = human_override_fn
        self._metrics_path = metrics_path

    def close(self) -> None:
        if isinstance(self._feature_provider, MachineAFeatureClient):
            self._feature_provider.close()

    def _current_collection_phase(self) -> str:
        if not hasattr(self._env, "current_phase_name"):
            return DEFAULT_COLLECTION_PHASE
        phase_name = str(self._env.current_phase_name())
        return phase_name.split(":", 1)[0]

    def _next_episode_id(self) -> int:
        replay_stats = self._replay_client.stats()
        return int(replay_stats.get("max_episode_id", -1)) + 1

    def run_episode(self, episode_id: int) -> dict[str, Any]:
        logger.info("EnvDriver episode=%s started", episode_id)
        observation = self._env.reset()
        collection_phase = self._current_collection_phase()
        done = False
        step_id = 0
        env_step_id = 0
        current_observation_idx = 0
        raw_episode = RawEpisodeTrace(
            episode_id=episode_id,
            chunk_len=self._rl_config.chunk_len,
            observations=[] if self._eval_actor_only else [observation],
            steps=[],
            chunks=[],
            policy_start_steps=[],
        )
        episode_success = 0
        fallback_count = 0
        intervention_count = 0
        dropped_transitions = 0
        source_counts = self._new_source_counts()
        actor_versions: list[int] = []
        start_time = time.time()

        while not done:
            plan_request_count = 0

            def _policy_planner(plan_observation: dict[str, Any], local_step: int) -> PolicyPlan:
                nonlocal fallback_count, intervention_count, plan_request_count
                current = normalize_feature_payload(
                    self._feature_provider.get_features(plan_observation),
                    self._rl_config,
                    observation=plan_observation,
                )
                current_features = _chunk_features_from_payload(current, self._rl_config)
                ref_chunk = current_features.ref_chunk
                refine = maybe_refine_chunk(
                    self._actor_client,
                    z_rl=current_features.z_rl,
                    proprio=current_features.proprio,
                    ref_chunk=ref_chunk,
                    request_id=f"{episode_id}:{step_id}:{plan_request_count}:{local_step}",
                    episode_id=episode_id,
                    step_id=env_step_id + local_step,
                    deterministic=self._env_config.actor_deterministic,
                    on_error_fallback=self._env_config.safe_fallback_to_ref,
                )
                plan_request_count += 1
                action_chunk = refine.refined_chunk
                source = refine.source
                actor_param_version = int(refine.actor_param_version)
                if refine.used_fallback:
                    fallback_count += 1
                    logger.warning(
                        "EnvDriver episode=%s chunk=%s plan=%s used fallback to ref_chunk error=%s",
                        episode_id,
                        step_id,
                        plan_request_count - 1,
                        refine.error,
                    )
                if self._env_config.enable_human_override and self._human_override_fn is not None:
                    maybe_human_chunk = self._human_override_fn(plan_observation, ref_chunk, action_chunk)
                    if maybe_human_chunk is not None:
                        action_chunk = np.asarray(maybe_human_chunk, dtype=np.float32)
                        source = int(TransitionSource.HUMAN)
                        actor_param_version = -1
                        intervention_count += 1
                        logger.warning(
                            "EnvDriver episode=%s chunk=%s plan=%s human override applied",
                            episode_id,
                            step_id,
                            plan_request_count - 1,
                        )
                if self._safe_action_filter is not None:
                    action_chunk = self._safe_action_filter(action_chunk)
                action_chunk = _coerce_action_chunk(
                    "planned action_chunk",
                    action_chunk,
                    expected_shape=ref_chunk.shape,
                )
                return PolicyPlan(
                    action_chunk=action_chunk,
                    ref_chunk=np.asarray(ref_chunk, dtype=np.float32),
                    source=int(source),
                    start_features=current_features,
                    actor_param_version=actor_param_version,
                )

            next_observation, rewards, done, info = self._execute_chunk(observation, _policy_planner)
            step_trace = info.get("step_trace") or []
            self._accumulate_rollout_trace(step_trace, source_counts, actor_versions)
            chunk_success = int(info.get("success", 0))
            if self._eval_actor_only:
                env_step_id += len(step_trace)
                observation = next_observation
                step_id += 1
                episode_success = int(info.get("success", episode_success))
                continue

            env_intervention_flag = bool(info.get("intervention_flag", False))
            if env_intervention_flag:
                intervention_count += 1
            if bool(info.get("drop_transition", False)):
                dropped_transitions += 1
                logger.warning(
                    "EnvDriver episode=%s chunk=%s marked drop_transition intervention=%s",
                    episode_id,
                    step_id,
                    env_intervention_flag,
                )

            source = int(info.get("source", int(TransitionSource.HUMAN)))
            trace_records = self._build_trace_records(
                step_trace,
                episode_id=episode_id,
                start_env_step_id=env_step_id,
                chunk_success=chunk_success,
                collection_phase=collection_phase,
            )
            env_step_id += len(trace_records)
            current_observation_idx = self._append_raw_chunk(
                raw_episode,
                observation_idx=current_observation_idx,
                trace_records=trace_records,
                chunk_step_id=step_id,
                chunk_source=source,
                collection_phase=collection_phase,
                done=done,
                success=chunk_success,
                drop_transition=bool(info.get("drop_transition", False)),
                start_features=info.get("chunk_start_features"),
                policy_anchor_offsets=info.get("policy_anchor_offsets") or [],
                policy_anchor_features=info.get("policy_anchor_features") or [],
            )
            observation = next_observation
            step_id += 1
            episode_success = int(info.get("success", episode_success))

        transitions_written = 0
        raw_episode_path: str | None = None
        replay_finalize_stats: dict[str, Any] = {}
        if not self._eval_actor_only:
            raw_episode.summary.update(
                {
                    "phase": self._env.current_phase_name() if hasattr(self._env, "current_phase_name") else None,
                    "collection_phase": collection_phase,
                    "fallback_count": fallback_count,
                    "intervention_count": intervention_count,
                    "dropped_transitions": dropped_transitions,
                    "policy_anchor_count": len(raw_episode.policy_start_steps),
                }
            )
            logger.info("EnvDriver episode=%s finalizing raw episode and replay...", episode_id)
            raw_episode_path = self._persist_raw_episode(raw_episode, episode_id=episode_id, started_at=start_time)
            transitions, replay_finalize_stats = self._build_episode_replay(raw_episode)
            if transitions:
                self._replay_client.add_transitions(transitions)
            transitions_written = len(transitions)
            logger.info(
                "EnvDriver episode=%s finalize done raw_steps=%s replay_transitions=%s cached_anchors=%s fetched_anchors=%s raw_episode=%s",
                episode_id,
                len(raw_episode.steps),
                transitions_written,
                replay_finalize_stats.get("cached_anchor_count", 0),
                replay_finalize_stats.get("fetched_anchor_count", 0),
                raw_episode_path,
            )
            logger.info("Episode %s finalize done. Ready for next episode. Press o to continue.", episode_id)
        summary = {
            "episode_id": episode_id,
            "num_chunk_steps": step_id,
            "success": episode_success,
            "fallback_count": fallback_count,
            "intervention_count": intervention_count,
            "transitions_written": transitions_written,
            "dropped_transitions": dropped_transitions,
            "eval_actor_only": self._eval_actor_only,
            "actor_deterministic": self._env_config.actor_deterministic,
            "collection_phase": collection_phase,
            "duration_sec": time.time() - start_time,
        }
        summary.update(self._summarize_rollout_trace(source_counts, actor_versions))
        if raw_episode_path is not None:
            summary["raw_episode_path"] = raw_episode_path
            summary.update(replay_finalize_stats)
        if hasattr(self._env, "current_phase_name"):
            summary["phase"] = self._env.current_phase_name()
        if self._metrics_path is not None:
            append_jsonl(self._metrics_path, summary)
        logger.info(
            "EnvDriver episode=%s finished chunk_steps=%s transitions_written=%s dropped=%s success=%s fallback_count=%s intervention_count=%s phase=%s",
            episode_id,
            step_id,
            transitions_written,
            dropped_transitions,
            episode_success,
            fallback_count,
            intervention_count,
            summary.get("phase"),
        )
        return summary

    @staticmethod
    def _new_source_counts() -> dict[int, int]:
        return {int(source): 0 for source in TransitionSource}

    @staticmethod
    def _accumulate_rollout_trace(
        step_trace: list[dict[str, Any]],
        source_counts: dict[int, int],
        actor_versions: list[int],
    ) -> None:
        for trace_step in step_trace:
            source = int(trace_step.get("source", int(TransitionSource.HUMAN)))
            source_counts[source] = source_counts.get(source, 0) + 1
            actor_version = int(trace_step.get("actor_param_version", -1))
            if actor_version >= 0:
                actor_versions.append(actor_version)

    @staticmethod
    def _summarize_rollout_trace(source_counts: dict[int, int], actor_versions: list[int]) -> dict[str, int]:
        if actor_versions:
            actor_version_start = actor_versions[0]
            actor_version_end = actor_versions[-1]
            actor_version_min = min(actor_versions)
            actor_version_max = max(actor_versions)
            actor_version_unique_count = len(set(actor_versions))
        else:
            actor_version_start = -1
            actor_version_end = -1
            actor_version_min = -1
            actor_version_max = -1
            actor_version_unique_count = 0
        return {
            "actor_version_start": actor_version_start,
            "actor_version_end": actor_version_end,
            "actor_version_min": actor_version_min,
            "actor_version_max": actor_version_max,
            "actor_version_unique_count": actor_version_unique_count,
            "rl_steps": source_counts.get(int(TransitionSource.RL), 0),
            "base_steps": source_counts.get(int(TransitionSource.BASE), 0),
            "human_steps": source_counts.get(int(TransitionSource.HUMAN), 0),
            "mixed_steps": source_counts.get(int(TransitionSource.MIXED), 0),
        }

    def _append_raw_chunk(
        self,
        raw_episode: RawEpisodeTrace,
        *,
        observation_idx: int,
        trace_records: list[StepTraceRecord],
        chunk_step_id: int,
        chunk_source: int,
        collection_phase: str,
        done: bool,
        success: int,
        drop_transition: bool,
        start_features: ChunkFeatures | None,
        policy_anchor_offsets: list[int],
        policy_anchor_features: list[ChunkFeatures],
    ) -> int:
        step_start = len(raw_episode.steps)
        current_observation_idx = observation_idx
        for trace_record in trace_records:
            next_observation_idx = len(raw_episode.observations)
            raw_episode.observations.append(trace_record.next_observation)
            raw_episode.steps.append(
                RawEpisodeStep(
                    observation_idx=current_observation_idx,
                    next_observation_idx=next_observation_idx,
                    action=np.asarray(trace_record.action, dtype=np.float32),
                    ref_action=np.asarray(trace_record.ref_action, dtype=np.float32),
                    reward=float(trace_record.reward),
                    done=bool(trace_record.done),
                    source=int(trace_record.source),
                    collection_phase=trace_record.collection_phase,
                    success=int(trace_record.success),
                    intervention_flag=bool(trace_record.human_controlled),
                    episode_id=trace_record.episode_id,
                    step_id=trace_record.env_step_id,
                    actor_param_version=int(trace_record.actor_param_version),
                )
            )
            current_observation_idx = next_observation_idx
        raw_episode.chunks.append(
            RawEpisodeChunk(
                episode_id=raw_episode.episode_id,
                chunk_step_id=int(chunk_step_id),
                observation_idx=int(observation_idx),
                step_start=step_start,
                step_stop=len(raw_episode.steps),
                source=int(chunk_source),
                collection_phase=collection_phase,
                done=bool(done),
                success=int(success),
                drop_transition=bool(drop_transition),
                start_z_rl=None if start_features is None else np.asarray(start_features.z_rl, dtype=np.float32),
                start_proprio=None if start_features is None else np.asarray(start_features.proprio, dtype=np.float32),
                start_ref_chunk=None
                if start_features is None
                else np.asarray(start_features.ref_chunk, dtype=np.float32),
            )
        )
        for local_offset, anchor_features in zip(policy_anchor_offsets, policy_anchor_features):
            absolute_start = step_start + int(local_offset)
            if step_start <= absolute_start < len(raw_episode.steps):
                raw_episode.policy_start_steps.append(absolute_start)
                self._record_feature_anchor(
                    raw_episode,
                    raw_episode.steps[absolute_start].observation_idx,
                    anchor_features,
                )
        return current_observation_idx

    @staticmethod
    def _record_feature_anchor(
        raw_episode: RawEpisodeTrace,
        observation_idx: int,
        features: ChunkFeatures,
    ) -> None:
        anchors = raw_episode.summary.setdefault("feature_anchors", {})
        payload = {
            "z_rl": np.asarray(features.z_rl, dtype=np.float32),
            "proprio": np.asarray(features.proprio, dtype=np.float32),
            "ref_chunk": np.asarray(features.ref_chunk, dtype=np.float32),
        }
        if features.source_ref_chunk is not None:
            payload["source_ref_chunk"] = np.asarray(features.source_ref_chunk, dtype=np.float32)
        anchors[int(observation_idx)] = payload

    def _persist_raw_episode(self, raw_episode: RawEpisodeTrace, *, episode_id: int, started_at: float) -> str:
        journal_path = self._replay_client.stats()["journal_path"]
        if journal_path is None:
            raise RuntimeError("Replay journal path is unavailable; cannot persist raw episode.")
        suffix = str(int(started_at))
        return save_raw_episode(raw_episode, raw_episode_path_for(journal_path, episode_id, suffix=suffix))

    def _build_episode_replay(self, raw_episode: RawEpisodeTrace) -> tuple[list[RLTTransition], dict[str, Any]]:
        feature_cache = self._seed_feature_cache(raw_episode)
        stats = {
            "cached_anchor_count": len(feature_cache),
            "fetched_anchor_count": 0,
            "raw_step_count": len(raw_episode.steps),
            "raw_chunk_count": len(raw_episode.chunks),
            "replay_mode": "dense" if self._env_config.step_trace_stride > 0 else "chunk",
            "step_trace_stride": int(self._env_config.step_trace_stride),
        }
        segments, raw_positions = self._collect_replay_segments(raw_episode)
        if self._env_config.step_trace_stride > 0:
            windows, terminal_window_added = self._build_dense_replay_windows(raw_episode, segments)
        else:
            windows, terminal_window_added = self._build_chunk_replay_windows(raw_episode, segments, raw_positions)
        transitions = self._build_replay_transitions(raw_episode, segments, windows, feature_cache, stats)
        stats["replay_window_count"] = len(windows)
        stats["terminal_window_added"] = int(terminal_window_added)
        stats["replay_transition_count"] = len(transitions)
        return transitions, stats

    def _seed_feature_cache(self, raw_episode: RawEpisodeTrace) -> dict[int, dict[str, Any]]:
        cache: dict[int, dict[str, Any]] = {}
        for chunk in raw_episode.chunks:
            if chunk.start_z_rl is None or chunk.start_proprio is None or chunk.start_ref_chunk is None:
                continue
            cache[int(chunk.observation_idx)] = _normalize_cached_feature_payload(
                {
                    "z_rl": chunk.start_z_rl,
                    "proprio": chunk.start_proprio,
                    "ref_chunk": chunk.start_ref_chunk,
                },
                self._rl_config,
            )
        for observation_idx, payload in raw_episode.summary.get("feature_anchors", {}).items():
            cache[int(observation_idx)] = _normalize_cached_feature_payload(payload, self._rl_config)
        return cache

    def _feature_payload_for_observation(
        self,
        raw_episode: RawEpisodeTrace,
        observation_idx: int,
        feature_cache: dict[int, dict[str, Any]],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        key = int(observation_idx)
        cached = feature_cache.get(key)
        if cached is not None:
            return _normalize_cached_feature_payload(cached, self._rl_config)
        payload = normalize_feature_payload(
            self._feature_provider.get_features(raw_episode.observations[key]),
            self._rl_config,
            observation=raw_episode.observations[key],
        )
        feature_cache[key] = payload
        stats["fetched_anchor_count"] += 1
        return payload

    def _collect_replay_segments(
        self,
        raw_episode: RawEpisodeTrace,
    ) -> tuple[list[ReplaySegment], dict[int, tuple[int, int]]]:
        segments: list[ReplaySegment] = []
        raw_positions: dict[int, tuple[int, int]] = {}
        current_segment: list[int] = []
        current_segment_id = 0
        for raw_chunk in raw_episode.chunks:
            if raw_chunk.drop_transition:
                if current_segment:
                    segments.append(ReplaySegment(raw_indices=current_segment))
                    current_segment = []
                current_segment_id = len(segments)
                continue
            if not current_segment:
                current_segment_id = len(segments)
            for raw_idx in range(raw_chunk.step_start, raw_chunk.step_stop):
                raw_positions[raw_idx] = (current_segment_id, len(current_segment))
                current_segment.append(raw_idx)
        if current_segment:
            segments.append(ReplaySegment(raw_indices=current_segment))
        return segments, raw_positions

    def _build_chunk_replay_windows(
        self,
        raw_episode: RawEpisodeTrace,
        segments: list[ReplaySegment],
        raw_positions: dict[int, tuple[int, int]],
    ) -> tuple[list[ReplayWindow], bool]:
        anchor_starts: set[int] = set()
        for raw_chunk in raw_episode.chunks:
            if raw_chunk.drop_transition:
                continue
            anchor_starts.add(int(raw_chunk.step_start))
        for raw_start in raw_episode.policy_start_steps:
            if int(raw_start) in raw_positions:
                anchor_starts.add(int(raw_start))

        windows: list[ReplayWindow] = []
        seen_starts: set[int] = set()
        for raw_start in sorted(anchor_starts):
            segment_id, start_offset = raw_positions[raw_start]
            if start_offset + self._rl_config.chunk_len > len(segments[segment_id].raw_indices):
                continue
            windows.append(ReplayWindow(segment_id=segment_id, start_offset=start_offset))
            seen_starts.add(raw_start)
        terminal_window_added = self._append_terminal_window(raw_episode, segments, windows, seen_starts)
        return windows, terminal_window_added

    def _build_dense_replay_windows(
        self,
        raw_episode: RawEpisodeTrace,
        segments: list[ReplaySegment],
    ) -> tuple[list[ReplayWindow], bool]:
        stride = int(self._env_config.step_trace_stride)
        windows: list[ReplayWindow] = []
        seen_starts: set[int] = set()
        for segment_id, segment in enumerate(segments):
            max_start = len(segment.raw_indices) - self._rl_config.chunk_len
            if max_start < 0:
                continue
            for start_offset in range(0, max_start + 1, stride):
                windows.append(ReplayWindow(segment_id=segment_id, start_offset=start_offset))
                seen_starts.add(segment.raw_indices[start_offset])
        terminal_window_added = self._append_terminal_window(raw_episode, segments, windows, seen_starts)
        return windows, terminal_window_added

    def _append_terminal_window(
        self,
        raw_episode: RawEpisodeTrace | None,
        segments: list[ReplaySegment],
        windows: list[ReplayWindow],
        seen_starts: set[int],
    ) -> bool:
        if not segments:
            return False
        last_segment = segments[-1].raw_indices
        if len(last_segment) < self._rl_config.chunk_len:
            return False
        if raw_episode is not None:
            last_step = raw_episode.steps[last_segment[-1]]
            if not last_step.done:
                return False
        terminal_start_offset = len(last_segment) - self._rl_config.chunk_len
        terminal_raw_start = last_segment[terminal_start_offset]
        if terminal_raw_start in seen_starts:
            return False
        windows.append(ReplayWindow(segment_id=len(segments) - 1, start_offset=terminal_start_offset))
        seen_starts.add(terminal_raw_start)
        return True

    def _prefetch_features_batch(
        self,
        raw_episode: RawEpisodeTrace,
        segments: list[ReplaySegment],
        windows: list[ReplayWindow],
        feature_cache: dict[int, dict[str, Any]],
        stats: dict[str, Any],
    ) -> None:
        """Collect all uncached observation indices and fetch them in fixed micro-batches."""
        needed: dict[int, dict[str, Any]] = {}  # observation_idx -> observation
        for window in windows:
            raw_indices = segments[window.segment_id].raw_indices[
                window.start_offset : window.start_offset + self._rl_config.chunk_len
            ]
            if not raw_indices:
                continue
            first_obs_idx = int(raw_episode.steps[raw_indices[0]].observation_idx)
            last_obs_idx = int(raw_episode.steps[raw_indices[-1]].next_observation_idx)
            for obs_idx in (first_obs_idx, last_obs_idx):
                if obs_idx not in feature_cache and obs_idx not in needed:
                    if obs_idx < len(raw_episode.observations):
                        needed[obs_idx] = raw_episode.observations[obs_idx]

        if not needed:
            return

        if hasattr(self._feature_provider, "get_features_batch"):
            sorted_indices = sorted(needed.keys())
            observations = [needed[idx] for idx in sorted_indices]
            micro_batch_size = int(self._env_config.replay_feature_batch_size)
            if micro_batch_size < 1:
                raise ValueError("env_driver.replay_feature_batch_size must be >= 1.")
            logger.info(
                "Prefetching %d features via batch request (cached=%d micro_batch_size=%d)",
                len(observations),
                len(feature_cache),
                micro_batch_size,
            )
            batch_start = time.time()
            results = []
            chunk_times_ms = []
            num_requests = 0
            for start in range(0, len(observations), micro_batch_size):
                chunk_observations = observations[start : start + micro_batch_size]
                chunk_start = time.time()
                chunk_results = self._feature_provider.get_features_batch(chunk_observations)
                chunk_time_ms = (time.time() - chunk_start) * 1000
                results.extend(chunk_results)
                chunk_times_ms.append(chunk_time_ms)
                num_requests += 1
                logger.info(
                    "Batch prefetch chunk %d size=%d done in %.1fms",
                    num_requests,
                    len(chunk_observations),
                    chunk_time_ms,
                )
            batch_time = time.time() - batch_start
            for idx, result in zip(sorted_indices, results, strict=True):
                feature_cache[idx] = normalize_feature_payload(result, self._rl_config, observation=needed[idx])
            stats["batch_prefetch_count"] = len(observations)
            stats["batch_prefetch_num_requests"] = num_requests
            stats["batch_prefetch_micro_batch_size"] = micro_batch_size
            stats["batch_prefetch_chunk_times_ms"] = chunk_times_ms
            stats["batch_prefetch_time_ms"] = batch_time * 1000
            stats["fetched_anchor_count"] += len(observations)
            logger.info(
                "Batch prefetch done: %d features in %.1fms (%.1fms/sample, requests=%d)",
                len(observations),
                batch_time * 1000,
                batch_time / len(observations) * 1000 if observations else 0,
                num_requests,
            )
        else:
            # Fallback: sequential fetch
            for obs_idx, obs in needed.items():
                payload = normalize_feature_payload(
                    self._feature_provider.get_features(obs),
                    self._rl_config,
                    observation=obs,
                )
                feature_cache[obs_idx] = payload
                stats["fetched_anchor_count"] += 1

    def _build_replay_transitions(
        self,
        raw_episode: RawEpisodeTrace,
        segments: list[ReplaySegment],
        windows: list[ReplayWindow],
        feature_cache: dict[int, dict[str, Any]],
        stats: dict[str, Any],
    ) -> list[RLTTransition]:
        if self._env_config.step_trace_stride > 0:
            stats["feature_prefetch_mode"] = "micro_batch"
            self._prefetch_features_batch(raw_episode, segments, windows, feature_cache, stats)
        else:
            stats["feature_prefetch_mode"] = "on_demand_single"

        transitions: list[RLTTransition] = []
        for window in windows:
            transitions.append(
                self._build_transition_from_window(
                    raw_episode,
                    segments[window.segment_id],
                    window,
                    feature_cache,
                    stats,
                )
            )
        return transitions

    def _build_transition_from_window(
        self,
        raw_episode: RawEpisodeTrace,
        segment: ReplaySegment,
        window: ReplayWindow,
        feature_cache: dict[int, dict[str, Any]],
        stats: dict[str, Any],
    ) -> RLTTransition:
        raw_indices = segment.raw_indices[window.start_offset : window.start_offset + self._rl_config.chunk_len]
        window_steps = [raw_episode.steps[raw_idx] for raw_idx in raw_indices]
        first_step = window_steps[0]
        last_step = window_steps[-1]
        current_payload = self._feature_payload_for_observation(
            raw_episode,
            first_step.observation_idx,
            feature_cache,
            stats,
        )
        next_payload = self._feature_payload_for_observation(
            raw_episode,
            last_step.next_observation_idx,
            feature_cache,
            stats,
        )
        action_chunk = np.stack([np.asarray(step.action, dtype=np.float32) for step in window_steps], axis=0)
        rewards = np.asarray([float(step.reward) for step in window_steps], dtype=np.float32)
        source_chunk = np.asarray([int(step.source) for step in window_steps], dtype=np.uint8)
        source, intervention = self._resolve_window_source(window_steps)
        return RLTTransition(
            z_rl=np.asarray(current_payload["z_rl"], dtype=np.float32),
            proprio=np.asarray(current_payload["proprio"], dtype=np.float32),
            ref_chunk=np.asarray(current_payload["ref_chunk"], dtype=np.float32),
            action_chunk=action_chunk,
            rewards=rewards,
            done=bool(any(step.done for step in window_steps) or last_step.done),
            next_z_rl=np.asarray(next_payload["z_rl"], dtype=np.float32),
            next_proprio=np.asarray(next_payload["proprio"], dtype=np.float32),
            next_ref_chunk=np.asarray(next_payload["ref_chunk"], dtype=np.float32),
            source=source,
            source_chunk=source_chunk,
            collection_phase=first_step.collection_phase,
            success=int(last_step.success),
            intervention_flag=intervention,
            episode_id=int(first_step.episode_id),
            step_id=int(first_step.step_id),
        )

    @staticmethod
    def _resolve_window_source(window_steps: list[RawEpisodeStep]) -> tuple[int, bool]:
        intervention = any(step.intervention_flag for step in window_steps)
        source_values = {int(step.source) for step in window_steps}
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
        return int(window_steps[0].source), intervention

    def _build_trace_records(
        self,
        step_trace: list[dict[str, Any]],
        *,
        episode_id: int,
        start_env_step_id: int,
        chunk_success: int,
        collection_phase: str,
    ) -> list[StepTraceRecord]:
        records: list[StepTraceRecord] = []
        for offset, trace_step in enumerate(step_trace):
            done = bool(trace_step.get("done", False))
            human_controlled = bool(trace_step.get("human_controlled", False))
            action = _coerce_action_vector(
                "step trace action",
                trace_step["action"],
                action_dim=self._rl_config.action_dim,
            )
            ref_action = _coerce_action_vector(
                "step trace ref_action",
                trace_step["ref_action"],
                action_dim=self._rl_config.action_dim,
            )
            records.append(
                StepTraceRecord(
                    observation=trace_step["observation"],
                    action=action,
                    ref_action=ref_action,
                    reward=float(trace_step["reward"]),
                    next_observation=trace_step["next_observation"],
                    source=int(trace_step.get("source", int(TransitionSource.HUMAN))),
                    collection_phase=collection_phase,
                    human_controlled=human_controlled,
                    done=done,
                    success=int(chunk_success if done else 0),
                    episode_id=episode_id,
                    env_step_id=start_env_step_id + offset,
                    actor_param_version=int(trace_step.get("actor_param_version", -1)),
                )
            )
        return records

    def run_forever(self, *, num_episodes: int | None = None) -> None:
        logger.debug("EnvDriver entering rollout loop num_episodes=%s", num_episodes)
        episode_id = self._next_episode_id()
        session_episode_count = 0
        logger.info("EnvDriver resuming episode numbering from episode_id=%s", episode_id)
        try:
            while num_episodes is None or session_episode_count < num_episodes:
                self.run_episode(episode_id)
                episode_id += 1
                session_episode_count += 1
        finally:
            self.close()
        logger.debug("EnvDriver rollout loop completed num_episodes=%s", num_episodes)

    def _execute_chunk(
        self,
        observation: dict[str, Any],
        policy_planner: Callable[[dict[str, Any], int], PolicyPlan],
    ) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
        if hasattr(self._env, "execute_chunk"):
            next_obs, rewards, done, info = self._env.execute_chunk(
                control_hz=self._env_config.control_frequency_hz,
                policy_planner=policy_planner,
            )
            return next_obs, list(rewards), bool(done), dict(info)

        plan = policy_planner(observation, 0)
        action_chunk = np.asarray(plan.action_chunk, dtype=np.float32)
        rewards: list[float] = []
        done = False
        info: dict[str, Any] = {}
        next_obs: dict[str, Any] | None = None
        for action in action_chunk[: self._env_config.chunk_exec_horizon]:
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            if done:
                break
        if next_obs is None:
            raise RuntimeError("Environment did not produce a next observation.")
        return next_obs, rewards, done, dict(info)
