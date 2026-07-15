from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import os
from pathlib import Path
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class RLTOnlineRLConfig:
    """Algorithm-only configuration for chunk-level online RL."""

    action_dim: int = 7
    chunk_len: int = 10
    z_dim: int = 2048
    proprio_dim: int = 7
    action_representation: Literal["abs", "delta_chunk"] = "abs"
    action_norm_stats_path: str | None = None
    # Actor/critic command-space fingerprint. Nero uses the real 19D EEF+hand
    # command, independently of Machine A's complete 26D VLA reference.
    action_layout_hash: str | None = None
    proprio_layout_hash: str | None = None
    # Optional explicit Machine-A reference projection. When configured, the
    # source payload must match this exact dimension/hash and is projected by
    # reference_action_indices before it reaches actor, critic, replay, or env.
    reference_action_dim: int | None = None
    reference_action_layout_hash: str | None = None
    reference_action_indices: tuple[int, ...] | None = None
    rot6d_convention: str | None = None
    # Action channels converted between absolute and delta representations and
    # used by the optional temporal-delta consistency loss. ``None`` preserves
    # the historical Agilex behavior: the first min(6, action_dim) channels.
    delta_action_indices: tuple[int, ...] | None = None

    gamma: float = 0.99
    fixed_std: float = 0.05
    reference_dropout_prob: float = 0.5
    warmup_bc_weight: float = 1.0
    warmup_q_weight: float = 1.0
    online_bc_weight: float = 1.0
    online_q_weight: float = 1.0
    delta_weight: float = 0.0

    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    target_tau: float = 5e-3
    actor_update_period: int = 2

    warmup_min_size: int = 1_000
    warmup_post_collect_updates: int | None = None
    freeze_after_warmup: bool = False
    # The paper reports a high update-to-data ratio of 5.
    grad_updates_per_cycle: int = 5

    def __post_init__(self) -> None:
        for name in ("action_dim", "chunk_len", "z_dim", "proprio_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        projection_values = (
            self.reference_action_dim,
            self.reference_action_layout_hash,
            self.reference_action_indices,
        )
        if any(value is not None for value in projection_values):
            if any(value is None for value in projection_values) or self.action_layout_hash is None:
                raise ValueError(
                    "reference projection requires reference_action_dim, "
                    "reference_action_layout_hash, reference_action_indices, and action_layout_hash"
                )
            assert self.reference_action_dim is not None
            assert self.reference_action_indices is not None
            if len(self.reference_action_indices) != self.action_dim:
                raise ValueError(f"reference_action_indices must contain exactly action_dim={self.action_dim} entries")
            if len(set(self.reference_action_indices)) != len(self.reference_action_indices):
                raise ValueError("reference_action_indices must not contain duplicates")
            if any(index < 0 or index >= self.reference_action_dim for index in self.reference_action_indices):
                raise ValueError(
                    f"reference_action_indices must be in range for reference_action_dim={self.reference_action_dim}"
                )


@dataclasses.dataclass(frozen=True)
class ActorServiceConfig:
    """Configuration for B1 actor_service."""

    pull_params_interval_sec: float = 0.25
    rpc_timeout_sec: float = 0.2
    bind_host: str = "127.0.0.1"
    port: int = 9101
    snapshot_path: str = "./artifacts/rlt/actor_snapshot.pkl"
    xla_mem_fraction: float = 0.10
    xla_preallocate: bool = False


@dataclasses.dataclass(frozen=True)
class LearnerServiceConfig:
    """Configuration for B2 learner_service."""

    push_actor_interval_steps: int = 50
    checkpoint_interval_steps: int = 1_000
    sample_batch_size: int = 256
    checkpoint_dir: str = "./artifacts/rlt/learner_checkpoints"
    actor_snapshot_path: str = "./artifacts/rlt/actor_snapshot.pkl"
    replay_url: str = "http://127.0.0.1:9102"
    poll_interval_sec: float = 0.01
    xla_mem_fraction: float = 0.75
    xla_preallocate: bool = False


@dataclasses.dataclass(frozen=True)
class ReplayConfig:
    """Configuration for B3 replay_manager."""

    capacity: int = 200_000
    journal_path: str = "./artifacts/rlt/replay_journal.pkl"
    bind_host: str = "127.0.0.1"
    port: int = 9102
    seed: int = 0
    sample_strategy: Literal["uniform", "stratified"] = "uniform"
    recent_episode_window: int = 20
    recent_online_ratio: float = 0.4
    warmup_demo_ratio: float = 0.3
    human_intervention_ratio: float = 0.2


@dataclasses.dataclass(frozen=True)
class EnvDriverConfig:
    """Configuration for B4 env_driver."""

    machine_a_ws_url: str = "ws://127.0.0.1:8000"
    # Generic/debug integrations may explicitly retain the historical unpinned
    # handshake. Nero production configs set this false and supply every value.
    machine_a_allow_unpinned_metadata: bool = True
    machine_a_expected_backend: str | None = None
    machine_a_expected_checkpoint_fingerprint: str | None = None
    machine_a_expected_cache_fingerprint: str | None = None
    machine_a_expected_encoder_artifact_sha256: str | None = None
    machine_a_expected_vlm_content_fingerprint: str | None = None
    actor_service_url: str = "http://127.0.0.1:9101"
    replay_service_url: str = "http://127.0.0.1:9102"
    task_mode: Literal["full_task", "critical_phase"] = "critical_phase"
    episode_start_control_mode: Literal["sticky", "policy", "human"] = "policy"
    full_task_reset_action: list[float] | None = None
    critical_phase_reset_action: list[float] | None = None
    actor_deterministic: bool = True
    chunk_exec_horizon: int = 10
    # RLT experiments run the robot at 50 Hz.
    control_frequency_hz: float = 50.0
    step_trace_stride: int = 0
    replay_feature_batch_size: int = 16
    enable_human_override: bool = False
    safe_fallback_to_ref: bool = True
    machine_a_connect_timeout_sec: float = 5.0
    machine_a_recv_timeout_sec: float = 5.0
    machine_a_retry_interval_sec: float = 0.5
    actor_request_timeout_sec: float = 1.0
    replay_request_timeout_sec: float = 30.0


@dataclasses.dataclass(frozen=True)
class MonitoringConfig:
    """Optional sidecar monitoring configuration."""

    enable_wandb: bool = False
    wandb_project: str = "rlt-online-rl"
    wandb_entity: str | None = None
    wandb_mode: Literal["offline", "online"] = "offline"
    wandb_run_name: str = "online-rl"
    wandb_dir: str = "./wandb"


@dataclasses.dataclass(frozen=True)
class OnlineRLSystemConfig:
    """Top-level configuration used by the orchestration script."""

    rl: RLTOnlineRLConfig = dataclasses.field(default_factory=RLTOnlineRLConfig)
    actor_service: ActorServiceConfig = dataclasses.field(default_factory=ActorServiceConfig)
    learner_service: LearnerServiceConfig = dataclasses.field(default_factory=LearnerServiceConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)
    env_driver: EnvDriverConfig = dataclasses.field(default_factory=EnvDriverConfig)
    monitoring: MonitoringConfig = dataclasses.field(default_factory=MonitoringConfig)

    role: Literal[
        "all",
        "actor_service",
        "learner_service",
        "replay_manager",
        "env_driver",
    ] = "all"
    local_debug_mode: bool = False


DEFAULT_CONFIG_FILENAME = "online_rl_config.yaml"

_ACTION_CONTRACT_FIELDS = (
    "action_dim",
    "proprio_dim",
    "chunk_len",
    "z_dim",
    "action_representation",
    "action_layout_hash",
    "proprio_layout_hash",
    "reference_action_dim",
    "reference_action_layout_hash",
    "reference_action_indices",
    "rot6d_convention",
    "delta_action_indices",
    "actor_hidden_dim",
    "actor_num_layers",
    "critic_hidden_dim",
    "critic_num_layers",
    "fixed_std",
)


def assert_action_contract_matches(
    saved_config: Mapping[str, Any] | None,
    current: RLTOnlineRLConfig,
    *,
    context: str,
) -> None:
    """Reject actor/critic artifacts created for a different tensor contract."""

    if not isinstance(saved_config, Mapping):
        raise ValueError(f"{context} does not contain an rl_config action contract")
    mismatches = []
    for name in _ACTION_CONTRACT_FIELDS:
        saved = saved_config.get(name)
        expected = getattr(current, name)
        if name in {"reference_action_indices", "delta_action_indices"}:
            saved = None if saved is None else tuple(saved)
        if saved != expected:
            mismatches.append(f"{name}: saved={saved!r} current={expected!r}")
    if mismatches:
        raise ValueError(f"{context} action contract mismatch: " + "; ".join(mismatches))


def _resolve_path_from_config(path_value: str | None, config_path: str, *, require_exists: bool) -> str | None:
    if path_value is None:
        return None
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    config_resolved = (Path(config_path).resolve().parent / candidate).resolve()
    project_resolved = (PROJECT_ROOT / candidate).resolve()
    if config_resolved.exists():
        return str(config_resolved)
    if project_resolved.exists():
        return str(project_resolved)
    if require_exists:
        return path_value
    if candidate.parts and candidate.parts[0] in {".", ".."}:
        return str(config_resolved)
    return str(project_resolved)


def _relativize_path_for_config(path_value: str | None, target_config_path: str) -> str | None:
    if path_value is None:
        return None
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        return path_value
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return os.path.relpath(str(candidate), start=os.path.dirname(os.path.abspath(target_config_path)))
    return os.path.relpath(str(candidate), start=str(PROJECT_ROOT))


def resolve_rl_config_paths(
    rl_config: RLTOnlineRLConfig,
    anchor_path: str,
    *,
    require_exists: bool = True,
) -> RLTOnlineRLConfig:
    return dataclasses.replace(
        rl_config,
        action_norm_stats_path=_resolve_path_from_config(
            rl_config.action_norm_stats_path, anchor_path, require_exists=require_exists
        ),
    )


def relativize_rl_config_paths(
    rl_config: RLTOnlineRLConfig,
    anchor_path: str,
) -> RLTOnlineRLConfig:
    return dataclasses.replace(
        rl_config,
        action_norm_stats_path=_relativize_path_for_config(rl_config.action_norm_stats_path, anchor_path),
    )


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _validate_mapping_keys(name: str, data: Mapping[str, Any], allowed_keys: set[str]) -> None:
    unknown = sorted(set(data) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown keys in {name}: {unknown}")


def _coerce_scalar_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        for arg in non_none_args:
            try:
                return _coerce_scalar_value(value, arg)
            except (TypeError, ValueError):
                continue
        return value
    if origin is Literal:
        return value
    if origin is tuple:
        args = get_args(annotation)
        item_type = args[0] if args else Any
        return tuple(_coerce_scalar_value(item, item_type) for item in value)
    if annotation is float:
        return float(value)
    if annotation is int:
        if isinstance(value, bool):
            raise TypeError("bool is not a valid int config value")
        return int(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        raise TypeError(f"Cannot coerce {value!r} to bool")
    return value


def _dataclass_from_mapping(cls: type[Any], data: Mapping[str, Any]) -> Any:
    default_value = cls()
    allowed_keys = {field.name for field in dataclasses.fields(default_value)}
    _validate_mapping_keys(cls.__name__, data, allowed_keys)
    field_types = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(default_value):
        value = data.get(field.name, getattr(default_value, field.name))
        default_field_value = getattr(default_value, field.name)
        if dataclasses.is_dataclass(default_field_value):
            if not isinstance(value, Mapping):
                raise TypeError(f"{cls.__name__}.{field.name} must be a mapping, got {type(value).__name__}")
            kwargs[field.name] = _dataclass_from_mapping(type(default_field_value), value)
        else:
            kwargs[field.name] = _coerce_scalar_value(value, field_types.get(field.name, field.type))
    return cls(**kwargs)


def split_system_config(system: OnlineRLSystemConfig) -> dict[str, Any]:
    return {
        "experiment": {
            "rl": dataclasses.asdict(system.rl),
        },
        "runtime": {
            "actor_service": dataclasses.asdict(system.actor_service),
            "learner_service": dataclasses.asdict(system.learner_service),
            "replay": dataclasses.asdict(system.replay),
            "env_driver": dataclasses.asdict(system.env_driver),
            "monitoring": dataclasses.asdict(system.monitoring),
            "role": system.role,
            "local_debug_mode": system.local_debug_mode,
        },
    }


def flatten_grouped_system_config(grouped_config: Mapping[str, Any]) -> dict[str, Any]:
    default_system = OnlineRLSystemConfig()
    top_level_keys = set(grouped_config)
    if top_level_keys & {"experiment", "runtime"}:
        _validate_mapping_keys("grouped config", grouped_config, {"experiment", "runtime"})
        experiment = grouped_config.get("experiment", {})
        runtime = grouped_config.get("runtime", {})
        if not isinstance(experiment, Mapping):
            raise TypeError(f"experiment must be a mapping, got {type(experiment).__name__}")
        if not isinstance(runtime, Mapping):
            raise TypeError(f"runtime must be a mapping, got {type(runtime).__name__}")
        _validate_mapping_keys("experiment", experiment, {"rl"})
        _validate_mapping_keys(
            "runtime",
            runtime,
            {"actor_service", "learner_service", "replay", "env_driver", "monitoring", "role", "local_debug_mode"},
        )
        return {
            "rl": dict(experiment.get("rl", {})),
            "actor_service": dict(runtime.get("actor_service", {})),
            "learner_service": dict(runtime.get("learner_service", {})),
            "replay": dict(runtime.get("replay", {})),
            "env_driver": dict(runtime.get("env_driver", {})),
            "monitoring": dict(runtime.get("monitoring", {})),
            "role": runtime.get("role", default_system.role),
            "local_debug_mode": runtime.get("local_debug_mode", default_system.local_debug_mode),
        }

    # Backward-compatible flat layout.
    _validate_mapping_keys(
        "flat config",
        grouped_config,
        {"rl", "actor_service", "learner_service", "replay", "env_driver", "monitoring", "role", "local_debug_mode"},
    )
    return {
        "rl": dict(grouped_config.get("rl", {})),
        "actor_service": dict(grouped_config.get("actor_service", {})),
        "learner_service": dict(grouped_config.get("learner_service", {})),
        "replay": dict(grouped_config.get("replay", {})),
        "env_driver": dict(grouped_config.get("env_driver", {})),
        "monitoring": dict(grouped_config.get("monitoring", {})),
        "role": grouped_config.get("role", default_system.role),
        "local_debug_mode": grouped_config.get("local_debug_mode", default_system.local_debug_mode),
    }


def system_config_from_mapping(data: Mapping[str, Any]) -> OnlineRLSystemConfig:
    flat = flatten_grouped_system_config(data)
    default_system = OnlineRLSystemConfig()
    return OnlineRLSystemConfig(
        rl=_dataclass_from_mapping(RLTOnlineRLConfig, flat["rl"]),
        actor_service=_dataclass_from_mapping(ActorServiceConfig, flat["actor_service"]),
        learner_service=_dataclass_from_mapping(LearnerServiceConfig, flat["learner_service"]),
        replay=_dataclass_from_mapping(ReplayConfig, flat["replay"]),
        env_driver=_dataclass_from_mapping(EnvDriverConfig, flat["env_driver"]),
        monitoring=_dataclass_from_mapping(MonitoringConfig, flat["monitoring"]),
        role=flat.get("role", default_system.role),
        local_debug_mode=flat.get("local_debug_mode", default_system.local_debug_mode),
    )


def load_system_config_yaml(
    path: str,
    *,
    base: OnlineRLSystemConfig | None = None,
) -> OnlineRLSystemConfig:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Config file {path} must contain a mapping at the top level.")
    merged = split_system_config(base or OnlineRLSystemConfig())
    _deep_update(merged, payload)
    system = system_config_from_mapping(merged)
    system = dataclasses.replace(
        system,
        rl=resolve_rl_config_paths(system.rl, path, require_exists=True),
        actor_service=dataclasses.replace(
            system.actor_service,
            snapshot_path=_resolve_path_from_config(system.actor_service.snapshot_path, path, require_exists=True),
        ),
        learner_service=dataclasses.replace(
            system.learner_service,
            checkpoint_dir=_resolve_path_from_config(system.learner_service.checkpoint_dir, path, require_exists=True),
            actor_snapshot_path=_resolve_path_from_config(
                system.learner_service.actor_snapshot_path, path, require_exists=True
            ),
        ),
        replay=dataclasses.replace(
            system.replay,
            journal_path=_resolve_path_from_config(system.replay.journal_path, path, require_exists=True),
        ),
        monitoring=dataclasses.replace(
            system.monitoring,
            wandb_dir=_resolve_path_from_config(system.monitoring.wandb_dir, path, require_exists=True),
        ),
    )
    return system


def save_system_config_yaml(system: OnlineRLSystemConfig, path: str) -> str:
    system = dataclasses.replace(
        system,
        rl=relativize_rl_config_paths(system.rl, path),
        actor_service=dataclasses.replace(
            system.actor_service,
            snapshot_path=_relativize_path_for_config(system.actor_service.snapshot_path, path),
        ),
        learner_service=dataclasses.replace(
            system.learner_service,
            checkpoint_dir=_relativize_path_for_config(system.learner_service.checkpoint_dir, path),
            actor_snapshot_path=_relativize_path_for_config(system.learner_service.actor_snapshot_path, path),
        ),
        replay=dataclasses.replace(
            system.replay,
            journal_path=_relativize_path_for_config(system.replay.journal_path, path),
        ),
        monitoring=dataclasses.replace(
            system.monitoring,
            wandb_dir=_relativize_path_for_config(system.monitoring.wandb_dir, path),
        ),
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(split_system_config(system), f, sort_keys=False)
    return path


def default_resolved_config_path(system: OnlineRLSystemConfig) -> str:
    return os.path.join(system.learner_service.checkpoint_dir, DEFAULT_CONFIG_FILENAME)
