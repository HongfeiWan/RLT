from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import ActorServiceConfig
from rlt_online_rl.config import LearnerServiceConfig
from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.config import save_system_config_yaml


def _load_run_online_rl_module():
    script_path = ROOT / "scripts" / "run_online_rl.py"
    spec = importlib.util.spec_from_file_location("run_online_rl_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_yaml_roundtrip_uses_experiment_and_runtime_sections(tmp_path) -> None:
    system = OnlineRLSystemConfig(
        rl=RLTOnlineRLConfig(
            chunk_len=12,
            z_dim=384,
            actor_hidden_dim=128,
            warmup_post_collect_updates=3000,
            freeze_after_warmup=True,
        ),
        actor_service=ActorServiceConfig(port=9201, snapshot_path=str(tmp_path / "actor.pkl")),
        learner_service=LearnerServiceConfig(checkpoint_dir=str(tmp_path / "ckpts")),
        role="learner_service",
        local_debug_mode=True,
    )
    path = tmp_path / "online_rl.yaml"
    save_system_config_yaml(system, str(path))

    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    assert set(payload) == {"experiment", "runtime"}
    assert payload["experiment"]["rl"]["chunk_len"] == 12
    assert payload["experiment"]["rl"]["warmup_post_collect_updates"] == 3000
    assert payload["experiment"]["rl"]["freeze_after_warmup"] is True
    assert payload["runtime"]["actor_service"]["port"] == 9201

    restored = load_system_config_yaml(str(path))
    assert restored.rl.chunk_len == 12
    assert restored.rl.z_dim == 384
    assert restored.rl.warmup_post_collect_updates == 3000
    assert restored.rl.freeze_after_warmup is True
    assert restored.actor_service.port == 9201
    assert restored.role == "learner_service"
    assert restored.local_debug_mode is True


def test_partial_yaml_merges_with_defaults(tmp_path) -> None:
    path = tmp_path / "partial.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment": {"rl": {"chunk_len": 6}},
                "runtime": {"actor_service": {"port": 9301}},
            },
            f,
            sort_keys=False,
        )

    restored = load_system_config_yaml(str(path))
    assert restored.rl.chunk_len == 6
    assert restored.actor_service.port == 9301
    assert restored.rl.action_dim == 7
    assert restored.learner_service.sample_batch_size == 256
    assert restored.replay.sample_strategy == "uniform"


def test_run_online_rl_config_file_provides_defaults_and_cli_can_override(tmp_path) -> None:
    module = _load_run_online_rl_module()
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment": {"rl": {"chunk_len": 12}},
                "runtime": {"actor_service": {"port": 9401}, "role": "actor_service"},
            },
            f,
            sort_keys=False,
        )

    args = module._parse_args(
        [
            "--config",
            str(config_path),
            "--system.rl.chunk-len",
            "8",
            "--system.role",
            "learner_service",
        ]
    )
    assert args.config == str(config_path)
    assert args.system.rl.chunk_len == 8
    assert args.system.actor_service.port == 9401
    assert args.system.role == "learner_service"


def test_replay_stratified_sampler_config_loads(tmp_path) -> None:
    path = tmp_path / "stratified.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "runtime": {
                    "replay": {
                        "sample_strategy": "stratified",
                        "recent_episode_window": 12,
                        "recent_online_ratio": 0.5,
                        "warmup_demo_ratio": 0.25,
                        "human_intervention_ratio": 0.15,
                    }
                }
            },
            f,
            sort_keys=False,
        )

    restored = load_system_config_yaml(str(path))
    assert restored.replay.sample_strategy == "stratified"
    assert restored.replay.recent_episode_window == 12
    assert restored.replay.recent_online_ratio == 0.5
    assert restored.replay.warmup_demo_ratio == 0.25
    assert restored.replay.human_intervention_ratio == 0.15


def test_resolved_config_is_saved_next_to_checkpoints(tmp_path) -> None:
    module = _load_run_online_rl_module()
    checkpoint_dir = tmp_path / "ckpts"
    args = module.Args(
        system=OnlineRLSystemConfig(
            learner_service=LearnerServiceConfig(checkpoint_dir=str(checkpoint_dir)),
            role="learner_service",
        )
    )

    path = module._maybe_save_resolved_config(args)
    assert path is not None
    assert Path(path).exists()

    restored = load_system_config_yaml(path)
    assert restored.learner_service.checkpoint_dir == str(checkpoint_dir)
    assert restored.role == "learner_service"


def test_paper_aligned_defaults_are_exposed_in_system_config() -> None:
    system = OnlineRLSystemConfig()
    assert system.rl.z_dim == 2048
    assert system.rl.warmup_post_collect_updates is None
    assert system.rl.freeze_after_warmup is False
    assert system.rl.grad_updates_per_cycle == 5
    assert system.env_driver.control_frequency_hz == 50.0
    assert system.env_driver.replay_request_timeout_sec == 30.0


def test_reference_projection_config_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="reference projection requires"):
        RLTOnlineRLConfig(reference_action_dim=26)

    with pytest.raises(ValueError, match="exactly action_dim"):
        RLTOnlineRLConfig(
            action_dim=19,
            action_layout_hash="sha256:target",
            reference_action_dim=26,
            reference_action_layout_hash="sha256:source",
            reference_action_indices=tuple(range(18)),
        )


def test_groot_nero_example_projects_26d_reference_to_19d_action() -> None:
    path = ROOT / "configs" / "tasks" / "groot_nero" / "online_rl.example.yaml"

    system = load_system_config_yaml(str(path))

    assert system.rl.action_dim == 19
    assert system.rl.proprio_dim == 19
    assert system.rl.z_dim == 2048
    assert system.rl.chunk_len == 10
    assert system.rl.action_representation == "abs"
    assert system.rl.delta_action_indices == ()
    assert system.rl.action_norm_stats_path == "/REPLACE_WITH_EXPORTED/groot_online_action_stats.json"
    assert system.rl.action_layout_hash == "sha256:REPLACE_WITH_19D_ACTION_LAYOUT_HASH"
    assert system.rl.proprio_layout_hash == "sha256:REPLACE_WITH_PROPRIO_LAYOUT_HASH"
    assert system.rl.reference_action_dim == 26
    assert system.rl.reference_action_layout_hash == ("sha256:REPLACE_WITH_26D_REFERENCE_LAYOUT_HASH")
    assert system.rl.reference_action_indices == tuple(range(19))
    assert system.rl.rot6d_convention == "groot_row_major_first_two_rows"
    assert system.env_driver.machine_a_ws_url == "ws://MACHINE_A_IP:8000"
    assert system.env_driver.machine_a_allow_unpinned_metadata is False
    assert system.env_driver.machine_a_expected_backend == "groot-n1.7"
    assert system.env_driver.machine_a_expected_checkpoint_fingerprint.startswith("sha256:")
    assert system.env_driver.machine_a_expected_cache_fingerprint.startswith("sha256:")
    assert system.env_driver.machine_a_expected_encoder_artifact_sha256.startswith("sha256:")
    assert system.env_driver.machine_a_expected_vlm_content_fingerprint.startswith("sha256:")
    assert system.env_driver.enable_human_override is False
