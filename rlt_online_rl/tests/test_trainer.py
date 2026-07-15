from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk
from rlt_online_rl.config import LearnerServiceConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.trainer import LearnerService
from rlt_online_rl.trainer import build_actor_supervision_targets
from rlt_online_rl.trainer import init_train_state
from rlt_online_rl.trainer import prepare_replay_training_batch
from rlt_online_rl.trainer import soft_update_targets
from rlt_online_rl.trainer import train_step
from rlt_online_rl.trainer import update_critic


class FakeReplay:
    def __init__(self, batch: dict[str, np.ndarray]):
        self._batch = batch
        self._adds_total = batch["z_rl"].shape[0]

    def sample_batch(self, _batch_size: int) -> dict[str, np.ndarray]:
        return self._batch

    def stats(self) -> dict[str, int]:
        return {"size": self._batch["z_rl"].shape[0], "adds_total": self._adds_total}


def _config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_update_period=2,
        warmup_min_size=1,
        grad_updates_per_cycle=2,
    )


def _batch(cfg: RLTOnlineRLConfig, batch_size: int = 8) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.ones((batch_size, cfg.z_dim), dtype=np.float32),
        "proprio": np.ones((batch_size, cfg.proprio_dim), dtype=np.float32),
        "ref_chunk": np.ones((batch_size, cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        "action_chunk": np.ones((batch_size, cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        "rewards": np.ones((batch_size, cfg.chunk_len), dtype=np.float32),
        "done": np.zeros((batch_size,), dtype=np.float32),
        "next_z_rl": np.ones((batch_size, cfg.z_dim), dtype=np.float32),
        "next_proprio": np.ones((batch_size, cfg.proprio_dim), dtype=np.float32),
        "next_ref_chunk": np.ones((batch_size, cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        "source": np.zeros((batch_size,), dtype=np.uint8),
        "source_chunk": np.zeros((batch_size, cfg.chunk_len), dtype=np.uint8),
        "valid_mask": np.ones((batch_size, cfg.chunk_len), dtype=np.bool_),
        "reference_valid_mask": np.ones((batch_size, cfg.chunk_len), dtype=np.bool_),
        "next_reference_valid_mask": np.ones((batch_size, cfg.chunk_len), dtype=np.bool_),
        "success": np.zeros((batch_size,), dtype=np.int8),
        "intervention_flag": np.zeros((batch_size,), dtype=np.bool_),
        "episode_id": np.zeros((batch_size,), dtype=np.int32),
        "step_id": np.arange(batch_size, dtype=np.int32),
    }


def test_actor_supervision_contract_uses_executed_human_and_mixed_actions() -> None:
    action = jax.numpy.asarray([[[10.0], [20.0], [30.0], [40.0], [50.0]]])
    reference = jax.numpy.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    source = jax.numpy.asarray(
        [[
            int(TransitionSource.BASE),
            int(TransitionSource.RL),
            int(TransitionSource.HUMAN),
            int(TransitionSource.MIXED),
            int(TransitionSource.MIXED),
        ]]
    )
    valid_mask = jax.numpy.asarray([[True, True, True, True, False]])

    target, valid, policy, human = build_actor_supervision_targets(
        action,
        reference,
        source,
        valid_mask,
    )

    assert np.array_equal(np.asarray(target[..., 0]), np.array([[1.0, 2.0, 30.0, 40.0, 0.0]]))
    assert np.array_equal(np.asarray(valid), np.array([[True, True, True, True, False]]))
    assert np.array_equal(np.asarray(policy), np.array([[True, True, False, False, False]]))
    assert np.array_equal(np.asarray(human), np.array([[False, False, True, True, False]]))


def test_critic_update_depends_on_executed_action_not_current_reference() -> None:
    cfg = _config()
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(17))
    base_np = _batch(cfg, batch_size=3)
    base_np["done"] = np.ones((3,), dtype=np.bool_)
    base_np["next_reference_valid_mask"] = np.zeros((3, cfg.chunk_len), dtype=np.bool_)
    base_np["next_ref_chunk"] = np.zeros_like(base_np["next_ref_chunk"])

    reference_changed_np = {key: value.copy() for key, value in base_np.items()}
    reference_changed_np["ref_chunk"] = np.full_like(reference_changed_np["ref_chunk"], 123.0)
    action_changed_np = {key: value.copy() for key, value in base_np.items()}
    action_changed_np["action_chunk"][:, 0, 0] = 7.0

    base_batch = {key: jax.numpy.asarray(value) for key, value in base_np.items()}
    reference_changed = {key: jax.numpy.asarray(value) for key, value in reference_changed_np.items()}
    action_changed = {key: jax.numpy.asarray(value) for key, value in action_changed_np.items()}
    base_state, base_metrics = update_critic(state, base_batch, actor, critic, cfg)
    reference_state, reference_metrics = update_critic(state, reference_changed, actor, critic, cfg)
    action_state, action_metrics = update_critic(state, action_changed, actor, critic, cfg)

    assert np.isclose(float(base_metrics["critic_loss"]), float(reference_metrics["critic_loss"]))
    for base_leaf, reference_leaf in zip(
        jax.tree.leaves(base_state.critic_params),
        jax.tree.leaves(reference_state.critic_params),
        strict=True,
    ):
        assert np.allclose(np.asarray(base_leaf), np.asarray(reference_leaf))
    assert not np.isclose(float(base_metrics["critic_loss"]), float(action_metrics["critic_loss"]))
    assert any(
        not np.allclose(np.asarray(base_leaf), np.asarray(action_leaf))
        for base_leaf, action_leaf in zip(
            jax.tree.leaves(base_state.critic_params),
            jax.tree.leaves(action_state.critic_params),
            strict=True,
        )
    )


def test_prepare_replay_training_batch_enforces_action_and_mask_contract() -> None:
    cfg = _config()
    batch = _batch(cfg, batch_size=2)
    batch["valid_mask"][:, -1] = False
    batch["reference_valid_mask"][:, -1] = False
    batch["ref_chunk"][:, -1] = 0.0
    batch["action_chunk"][:, -1] = 0.0
    batch["rewards"][:, -1] = 0.0
    prepared = prepare_replay_training_batch(batch, cfg)
    assert prepared["action_chunk"].dtype == np.float32
    assert prepared["action_chunk"].shape == (2, cfg.chunk_len, cfg.action_dim)

    wrong_dim = {key: value.copy() for key, value in batch.items()}
    wrong_dim["action_chunk"] = np.zeros((2, cfg.chunk_len, cfg.action_dim + 1), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "action_chunk has shape"):
        prepare_replay_training_batch(wrong_dim, cfg)

    non_finite = {key: value.copy() for key, value in batch.items()}
    non_finite["action_chunk"][0, 0, 0] = np.inf
    with np.testing.assert_raises_regex(ValueError, "action_chunk contains non-finite"):
        prepare_replay_training_batch(non_finite, cfg)


def test_train_step_runs_and_actor_update_period_works() -> None:
    cfg = _config()
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(0))
    batch = {k: jax.numpy.asarray(v) for k, v in _batch(cfg).items()}
    state, metrics1 = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    assert int(metrics1["did_actor_update"]) == 0
    state, metrics2 = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    assert int(metrics2["did_actor_update"]) == 1


def test_train_step_bc_target_switches_to_human_actions() -> None:
    cfg = _config()
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(0))
    batch_np = _batch(cfg)
    batch_np["ref_chunk"] = np.zeros_like(batch_np["ref_chunk"])
    batch_np["action_chunk"] = np.ones_like(batch_np["action_chunk"])
    batch_np["source_chunk"] = np.full_like(batch_np["source_chunk"], int(TransitionSource.HUMAN), dtype=np.uint8)
    batch = {k: jax.numpy.asarray(v) for k, v in batch_np.items()}

    state, _ = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    _state, metrics = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)

    assert int(metrics["did_actor_update"]) == 1
    assert np.isclose(float(metrics["human_mask_ratio"]), 1.0)
    assert np.isclose(float(metrics["policy_mask_ratio"]), 0.0)
    assert np.isclose(float(metrics["bc_penalty"]), float(metrics["bc_human_penalty"]))
    assert np.isclose(float(metrics["bc_ref_penalty"]), 0.0)


def test_soft_update_changes_target_params() -> None:
    cfg = _config()
    state, _, _ = init_train_state(cfg, rng=jax.random.PRNGKey(1))
    updated = soft_update_targets(state.target_actor_params, jax.tree.map(lambda x: x + 1.0, state.actor_params), 0.5)
    leaves_old = jax.tree.leaves(state.target_actor_params)
    leaves_new = jax.tree.leaves(updated)
    assert any(not np.allclose(np.asarray(o), np.asarray(n)) for o, n in zip(leaves_old, leaves_new, strict=True))


def test_learner_service_exports_actor_snapshot(tmp_path) -> None:
    cfg = _config()
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=1,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, FakeReplay(_batch(cfg)))
    metrics = learner.train_once()
    assert metrics is not None
    metrics = learner.train_once()
    assert metrics is not None
    snapshot = learner.get_actor_snapshot()
    assert snapshot["version"] == 1
    assert (tmp_path / "actor.pkl").exists()
    history_path = tmp_path / "history" / "actor_v000001.pkl"
    assert history_path.exists()
    with history_path.open("rb") as f:
        history_payload = pickle.load(f)
    assert history_payload["version"] == 1
    assert history_payload["global_step"] == 2


def test_learner_service_stops_when_update_budget_is_consumed(tmp_path) -> None:
    cfg = _config()
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=100,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, FakeReplay(_batch(cfg, batch_size=3)))
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None


def test_learner_service_uses_fixed_warmup_budget_then_online_ratio(tmp_path) -> None:
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_update_period=2,
        warmup_min_size=1,
        warmup_post_collect_updates=3,
        grad_updates_per_cycle=5,
    )
    replay = FakeReplay(_batch(cfg, batch_size=3))
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=100,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, replay)

    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None

    replay._adds_total = 4
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None


def test_learner_service_freezes_after_warmup_when_configured(tmp_path) -> None:
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_update_period=2,
        warmup_min_size=1,
        warmup_post_collect_updates=3,
        freeze_after_warmup=True,
        grad_updates_per_cycle=5,
    )
    replay = FakeReplay(_batch(cfg, batch_size=3))
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=100,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, replay)

    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None

    replay._adds_total = 4
    assert learner.train_once() is None


def test_delta_chunk_normalization_broadcasts_batched_state0(tmp_path) -> None:
    stats_path = tmp_path / "norm_stats_delta.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {
                        "q01": [-1.0, -1.0, -1.0],
                        "q99": [1.0, 1.0, 1.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        proprio_dim=3,
        action_representation="delta_chunk",
        action_norm_stats_path=str(stats_path),
    )
    adapter = ActionRepresentationAdapter.from_config(cfg)
    assert adapter is not None
    chunk_abs = np.ones((8, cfg.chunk_len, cfg.action_dim), dtype=np.float32)
    state0 = np.full((8, cfg.proprio_dim), 0.5, dtype=np.float32)
    normalized = adapter.normalize_chunk(chunk_abs, state0)
    assert normalized.shape == chunk_abs.shape
    restored = np.asarray(
        jax.device_get(
            jax_denormalize_to_abs_chunk(
                jax.numpy.asarray(normalized),
                jax.numpy.asarray(state0),
                jax.numpy.asarray(adapter.stats.q01),
                jax.numpy.asarray(adapter.stats.q99),
                action_representation=cfg.action_representation,
            )
        ),
        dtype=np.float32,
    )
    assert np.allclose(restored, chunk_abs, atol=1e-5)
