from __future__ import annotations

import dataclasses
import http.client
import importlib
import logging
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any
from urllib import error as urllib_error

import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import default_resolved_config_path
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.config import save_system_config_yaml
from rlt_online_rl.runtime_logging import ensure_runtime_dirs
from rlt_online_rl.runtime_logging import infer_run_dir
from rlt_online_rl.runtime_logging import setup_process_logging


@dataclasses.dataclass
class Args:
    config: str | None = None
    resolved_config_path: str | None = None
    system: OnlineRLSystemConfig = dataclasses.field(default_factory=OnlineRLSystemConfig)
    env_factory: str | None = None
    human_override_factory: str | None = None
    num_episodes: int | None = None


class DummyFeatureProvider:
    def __init__(self, z_dim: int, proprio_dim: int, chunk_len: int, action_dim: int):
        self._z_dim = z_dim
        self._proprio_dim = proprio_dim
        self._chunk_len = chunk_len
        self._action_dim = action_dim

    def get_features(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(observation["state"], dtype=np.float32)
        z_rl = np.pad(state, (0, max(0, self._z_dim - state.shape[0])), mode="constant")[: self._z_dim]
        proprio = np.pad(state, (0, max(0, self._proprio_dim - state.shape[0])), mode="constant")[: self._proprio_dim]
        ref_action = np.tanh(state[: self._action_dim])
        ref_chunk = np.tile(ref_action[None, :], (self._chunk_len, 1))
        return {"z_rl": z_rl, "proprio": proprio, "ref_chunk": ref_chunk}


class DummyChunkEnv:
    def __init__(self, state_dim: int, action_dim: int):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._step = 0

    def reset(self) -> dict[str, Any]:
        self._step = 0
        return {"state": np.zeros((self._state_dim,), dtype=np.float32)}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        self._step += 1
        reward = float(-np.mean(np.square(action)))
        terminated = self._step >= 25
        next_obs = {"state": np.full((self._state_dim,), self._step, dtype=np.float32)}
        info = {"success": int(terminated and reward > -0.5)}
        return next_obs, reward, terminated, False, info


def _configure_xla(*, preallocate: bool, mem_fraction: float) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)


def _load_factory(path: str | None) -> Any | None:
    if path is None:
        return None
    module_name, attr_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _run_actor_service(system: OnlineRLSystemConfig) -> None:
    log_path = setup_process_logging("actor_service", system, console_level=logging.WARNING)
    logger = logging.getLogger("actor_service")
    _configure_xla(
        preallocate=system.actor_service.xla_preallocate,
        mem_fraction=system.actor_service.xla_mem_fraction,
    )
    from rlt_online_rl.inference import ActorService

    stop_event = threading.Event()

    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        logger.info("Received SIGTERM; stopping actor_service.")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    logger.debug(
        "Starting actor_service log=%s bind=%s:%s snapshot=%s",
        log_path,
        system.actor_service.bind_host,
        system.actor_service.port,
        system.actor_service.snapshot_path,
    )
    service = ActorService(system.rl, system.actor_service)
    service.serve_forever(stop_event=stop_event)
    logger.debug("actor_service stopped.")


def _run_replay_manager(system: OnlineRLSystemConfig) -> None:
    log_path = setup_process_logging("replay_manager", system, console_level=logging.WARNING)
    logger = logging.getLogger("replay_manager")
    from rlt_online_rl.replay import ReplayManager
    from rlt_online_rl.replay import ReplayTensorContract

    stop_event = threading.Event()

    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        logger.info("Received SIGTERM; stopping replay_manager.")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    logger.debug(
        "Starting replay_manager log=%s bind=%s:%s journal=%s",
        log_path,
        system.replay.bind_host,
        system.replay.port,
        system.replay.journal_path,
    )
    manager = ReplayManager(
        system.replay.capacity,
        journal_path=system.replay.journal_path,
        seed=system.replay.seed,
        metrics_path=str(ensure_runtime_dirs(system)["metrics_dir"] / "replay_stats.jsonl"),
        sample_strategy=system.replay.sample_strategy,
        recent_episode_window=system.replay.recent_episode_window,
        recent_online_ratio=system.replay.recent_online_ratio,
        warmup_demo_ratio=system.replay.warmup_demo_ratio,
        human_intervention_ratio=system.replay.human_intervention_ratio,
        tensor_contract=ReplayTensorContract(
            z_dim=system.rl.z_dim,
            proprio_dim=system.rl.proprio_dim,
            chunk_len=system.rl.chunk_len,
            action_dim=system.rl.action_dim,
        ),
    )
    manager.serve_forever(system.replay.bind_host, system.replay.port, stop_event=stop_event)
    logger.debug("replay_manager stopped.")


def _run_learner_service(system: OnlineRLSystemConfig) -> None:
    log_path = setup_process_logging("learner_service", system, console_level=logging.INFO)
    logger = logging.getLogger("learner_service")
    _configure_xla(
        preallocate=system.learner_service.xla_preallocate,
        mem_fraction=system.learner_service.xla_mem_fraction,
    )
    from rlt_online_rl.replay import ReplayClient
    from rlt_online_rl.trainer import LearnerService

    stop_event = threading.Event()

    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        logger.info("Received SIGTERM; stopping learner_service.")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    logger.debug(
        "Starting learner_service log=%s replay_url=%s checkpoint_dir=%s snapshot=%s",
        log_path,
        system.learner_service.replay_url,
        system.learner_service.checkpoint_dir,
        system.learner_service.actor_snapshot_path,
    )
    replay_client = ReplayClient(system.learner_service.replay_url)
    learner = LearnerService(
        system.rl,
        system.learner_service,
        replay_client,
        metrics_path=str(ensure_runtime_dirs(system)["metrics_dir"] / "learner_metrics.jsonl"),
    )
    try:
        learner.run_forever(stop_event=stop_event)
    finally:
        learner.flush_artifacts()
    logger.debug("learner_service stopped.")


def _run_env_driver(
    system: OnlineRLSystemConfig,
    *,
    env_factory: str | None,
    human_override_factory: str | None,
    num_episodes: int | None,
) -> None:
    log_path = setup_process_logging("env_driver", system, console_level=logging.INFO)
    logger = logging.getLogger("env_driver")
    from rlt_online_rl.inference import ActorClient
    from rlt_online_rl.inference import EnvDriver
    from rlt_online_rl.inference import MachineAFeatureClient
    from rlt_online_rl.replay import ReplayClient

    if system.local_debug_mode:
        env = DummyChunkEnv(system.rl.proprio_dim, system.rl.action_dim)
        feature_provider = DummyFeatureProvider(
            system.rl.z_dim,
            system.rl.proprio_dim,
            system.rl.chunk_len,
            system.rl.action_dim,
        )
    else:
        if env_factory is None:
            raise ValueError("env_factory is required when local_debug_mode=False.")
        env = _load_factory(env_factory)()
        feature_provider = MachineAFeatureClient(
            system.env_driver.machine_a_ws_url,
            connect_timeout_sec=system.env_driver.machine_a_connect_timeout_sec,
            recv_timeout_sec=system.env_driver.machine_a_recv_timeout_sec,
            retry_interval_sec=system.env_driver.machine_a_retry_interval_sec,
            allow_unpinned_metadata=system.env_driver.machine_a_allow_unpinned_metadata,
            expected_backend=system.env_driver.machine_a_expected_backend,
            expected_checkpoint_fingerprint=(
                system.env_driver.machine_a_expected_checkpoint_fingerprint
            ),
            expected_cache_fingerprint=system.env_driver.machine_a_expected_cache_fingerprint,
            expected_encoder_artifact_sha256=(
                system.env_driver.machine_a_expected_encoder_artifact_sha256
            ),
            expected_vlm_content_fingerprint=(
                system.env_driver.machine_a_expected_vlm_content_fingerprint
            ),
        )

    human_override_fn = _load_factory(human_override_factory)
    actor_client = ActorClient(
        system.env_driver.actor_service_url,
        timeout_sec=system.env_driver.actor_request_timeout_sec,
    )
    replay_client = ReplayClient(
        system.env_driver.replay_service_url,
        timeout_sec=system.env_driver.replay_request_timeout_sec,
    )
    logger.debug(
        "Starting env_driver log=%s local_debug=%s actor_service=%s replay_service=%s machine_a=%s num_episodes=%s",
        log_path,
        system.local_debug_mode,
        system.env_driver.actor_service_url,
        system.env_driver.replay_service_url,
        system.env_driver.machine_a_ws_url,
        num_episodes,
    )
    driver = EnvDriver(
        env,
        feature_provider,
        actor_client,
        replay_client,
        system.rl,
        system.env_driver,
        human_override_fn=human_override_fn,
        metrics_path=str(ensure_runtime_dirs(system)["metrics_dir"] / "rollout_metrics.jsonl"),
    )
    driver.run_forever(num_episodes=num_episodes)
    logger.debug("env_driver completed num_episodes=%s", num_episodes)


def _spawn_process(name: str, target: Any, *args: Any, **kwargs: Any) -> mp.Process:
    process = mp.Process(
        name=name,
        target=target,
        args=args,
        kwargs=kwargs,
        daemon=False,
    )
    process.start()
    return process


def _terminate_processes(processes: list[mp.Process], *, logger: logging.Logger, grace_sec: float = 5.0) -> None:
    alive_processes = [process for process in processes if process.is_alive()]
    for process in alive_processes:
        logger.info("Sending SIGTERM to %s pid=%s", process.name, process.pid)
        process.terminate()

    deadline = time.time() + grace_sec
    while time.time() < deadline and any(process.is_alive() for process in alive_processes):
        time.sleep(0.1)

    for process in alive_processes:
        if process.is_alive():
            logger.warning("Force killing %s pid=%s after %.1fs grace period", process.name, process.pid, grace_sec)
            process.kill()

    for process in processes:
        process.join(timeout=1.0)


def _terminate_process(process: mp.Process | None, *, logger: logging.Logger, grace_sec: float = 5.0) -> None:
    if process is None:
        return
    if process.exitcode is not None:
        process.join(timeout=1.0)
        return
    logger.info("Sending SIGTERM to %s pid=%s", process.name, process.pid)
    process.terminate()
    deadline = time.time() + grace_sec
    while time.time() < deadline and process.exitcode is None:
        time.sleep(0.1)
    if process.exitcode is None:
        logger.warning("Force killing %s pid=%s after %.1fs grace period", process.name, process.pid, grace_sec)
        process.kill()
    process.join(timeout=1.0)


def _wait_for_actor_service_ready(system: OnlineRLSystemConfig, *, logger: logging.Logger) -> None:
    from rlt_online_rl.inference import ActorClient

    client = ActorClient(
        system.env_driver.actor_service_url,
        timeout_sec=max(system.actor_service.rpc_timeout_sec, 0.5),
        max_retries=0,
    )
    deadline = time.time() + 20.0
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            version = client.get_actor_param_version()
            if version >= 0:
                logger.info("actor_service ready actor_param_version=%s", version)
                return
        except (RuntimeError, urllib_error.URLError, ConnectionError, http.client.HTTPException) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError("actor_service did not become ready before timeout") from last_error


def _peek_option(argv: list[str], flag: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value.")
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _parse_args(argv: list[str] | None = None) -> Args:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _peek_option(argv, "--config")
    default_args = Args()
    if config_path is not None:
        default_args = dataclasses.replace(
            default_args,
            config=config_path,
            system=load_system_config_yaml(config_path),
        )
    return tyro.cli(Args, args=argv, default=default_args)


def _maybe_save_resolved_config(args: Args) -> str | None:
    system = args.system
    role = system.role
    if args.resolved_config_path is None and role not in {"all", "learner_service"}:
        return None
    path = args.resolved_config_path or default_resolved_config_path(system)
    return save_system_config_yaml(system, path)


def main(args: Args) -> None:
    system = args.system
    role = system.role
    log_path = setup_process_logging("supervisor", system, console_level=logging.INFO)
    logger = logging.getLogger("supervisor")
    resolved_config_path = _maybe_save_resolved_config(args)
    runtime_dirs = ensure_runtime_dirs(system)
    logger.info(
        "Starting online RL role=%s run_dir=%s log=%s config=%s resolved_config=%s local_debug=%s num_episodes=%s",
        role,
        infer_run_dir(system),
        log_path,
        args.config,
        resolved_config_path,
        system.local_debug_mode,
        args.num_episodes,
    )
    logger.info(
        "Artifacts checkpoints=%s replay=%s actor_snapshot=%s logs=%s metrics=%s",
        system.learner_service.checkpoint_dir,
        system.replay.journal_path,
        system.actor_service.snapshot_path,
        runtime_dirs["log_dir"],
        runtime_dirs["metrics_dir"],
    )

    if role == "actor_service":
        _run_actor_service(system)
        return
    if role == "replay_manager":
        _run_replay_manager(system)
        return
    if role == "learner_service":
        _run_learner_service(system)
        return
    if role == "env_driver":
        _run_env_driver(
            system,
            env_factory=args.env_factory,
            human_override_factory=args.human_override_factory,
            num_episodes=args.num_episodes,
        )
        return

    processes: list[mp.Process] = []
    process_map: dict[str, mp.Process] = {}
    env_driver_completed = False
    try:
        processes.append(_spawn_process("replay_manager", _run_replay_manager, system))
        process_map["replay_manager"] = processes[-1]
        logger.info("Started replay_manager pid=%s", processes[-1].pid)
        time.sleep(0.5)
        processes.append(_spawn_process("learner_service", _run_learner_service, system))
        process_map["learner_service"] = processes[-1]
        logger.info("Started learner_service pid=%s", processes[-1].pid)
        time.sleep(0.5)
        processes.append(_spawn_process("actor_service", _run_actor_service, system))
        process_map["actor_service"] = processes[-1]
        logger.info("Started actor_service pid=%s", processes[-1].pid)
        _wait_for_actor_service_ready(system, logger=logger)
        time.sleep(0.5)
        if system.local_debug_mode or args.env_factory is not None:
            processes.append(
                _spawn_process(
                    "env_driver",
                    _run_env_driver,
                    system,
                    env_factory=args.env_factory,
                    human_override_factory=args.human_override_factory,
                    num_episodes=args.num_episodes,
                )
            )
            process_map["env_driver"] = processes[-1]
            logger.info("Started env_driver pid=%s", processes[-1].pid)
        while True:
            for process in processes:
                if process.exitcode is None:
                    continue
                if process.name == "env_driver" and process.exitcode == 0 and args.num_episodes is not None:
                    logger.info(
                        "env_driver completed successfully after num_episodes=%s; initiating shutdown.",
                        args.num_episodes,
                    )
                    env_driver_completed = True
                    return
                if process.exitcode != 0:
                    raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt; shutting down processes.")
    finally:
        if env_driver_completed:
            _terminate_process(process_map.get("learner_service"), logger=logger, grace_sec=15.0)
            _terminate_process(process_map.get("actor_service"), logger=logger)
            _terminate_process(process_map.get("replay_manager"), logger=logger)
            _terminate_process(process_map.get("env_driver"), logger=logger)
        else:
            _terminate_processes(processes, logger=logger)
        if env_driver_completed:
            logger.info("Supervisor shutdown complete after env_driver completion.")
        else:
            logger.info("Supervisor shutdown complete.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    mp.set_start_method("spawn", force=True)
    main(_parse_args())
