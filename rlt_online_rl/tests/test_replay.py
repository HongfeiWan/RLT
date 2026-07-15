from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import rlt_online_rl.replay as replay_module
from rlt_online_rl.replay import COLLECTION_PHASE_ONLINE
from rlt_online_rl.replay import COLLECTION_PHASE_WARMUP
from rlt_online_rl.replay import EpisodeStepRecord
from rlt_online_rl.replay import ReplayBuffer
from rlt_online_rl.replay import ReplayManager
from rlt_online_rl.replay import ReplayTensorContract
from rlt_online_rl.replay import RLTTransition
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.replay import build_chunk_transitions_from_episode
from rlt_online_rl.replay import build_terminal_aligned_chunk_transition


def _make_episode(
    length: int = 6,
    *,
    intervention_at: int | None = None,
    policy_source: int = int(TransitionSource.RL),
    collection_phase: str = "unknown",
    episode_id: int = 7,
) -> list[EpisodeStepRecord]:
    steps = []
    for idx in range(length):
        action = np.full((2,), idx + 1, dtype=np.float32)
        human_controlled = idx == intervention_at
        steps.append(
            EpisodeStepRecord(
                z_rl=np.full((4,), idx, dtype=np.float32),
                proprio=np.full((3,), idx, dtype=np.float32),
                ref_action=action.copy() if human_controlled else np.full((2,), idx + 0.1, dtype=np.float32),
                action=action,
                reward=float(idx),
                done=idx == length - 1,
                next_z_rl=np.full((4,), idx + 1, dtype=np.float32),
                next_proprio=np.full((3,), idx + 1, dtype=np.float32),
                source=int(TransitionSource.HUMAN) if human_controlled else policy_source,
                collection_phase=collection_phase,
                success=int(idx == length - 1),
                intervention_flag=human_controlled,
                episode_id=episode_id,
                step_id=idx,
            )
        )
    return steps


def test_stride2_builds_expected_chunk_transitions() -> None:
    transitions = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)
    assert len(transitions) == 3
    assert transitions[0].next_ref_chunk.shape == (3, 2)
    assert transitions[0].source_chunk.shape == (3,)
    assert transitions[1].step_id == 2


def test_intervention_replaces_only_intervened_reference_steps() -> None:
    transitions = build_chunk_transitions_from_episode(_make_episode(intervention_at=1), chunk_len=3, stride=2)
    first = transitions[0]
    assert first.intervention_flag
    assert np.allclose(first.ref_chunk[0], np.full((2,), 0.1, dtype=np.float32))
    assert np.allclose(first.ref_chunk[1], first.action_chunk[1])
    assert np.allclose(first.ref_chunk[2], np.full((2,), 2.1, dtype=np.float32))
    assert first.source == int(TransitionSource.MIXED)
    assert int(first.source_chunk[1]) == int(TransitionSource.HUMAN)


def test_base_and_human_steps_resolve_to_mixed() -> None:
    transitions = build_chunk_transitions_from_episode(
        _make_episode(intervention_at=1, policy_source=int(TransitionSource.BASE)),
        chunk_len=3,
        stride=2,
    )
    assert transitions[0].source == int(TransitionSource.MIXED)


def test_terminal_aligned_chunk_uses_last_full_window() -> None:
    transition = build_terminal_aligned_chunk_transition(_make_episode(length=7), chunk_len=3)
    assert transition is not None
    assert transition.step_id == 4
    assert transition.done
    assert transition.success == 1
    assert np.allclose(transition.action_chunk[:, 0], np.array([5.0, 6.0, 7.0], dtype=np.float32))


def test_replay_append_and_sample() -> None:
    transitions = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)
    replay = ReplayBuffer(capacity=16, seed=0)
    replay.extend(transitions)
    batch = replay.sample(2)
    assert batch["z_rl"].shape[0] == 2
    assert batch["next_ref_chunk"].shape[1:] == (3, 2)
    assert batch["source_chunk"].shape[1:] == (3,)
    assert batch["action_chunk"].dtype == np.float32
    assert batch["ref_chunk"].dtype == np.float32
    assert batch["next_ref_chunk"].dtype == np.float32
    assert batch["valid_mask"].dtype == np.bool_


def test_terminal_partial_chunk_has_explicit_prefix_masks() -> None:
    transitions = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)
    terminal = transitions[-1]

    assert terminal.done
    assert np.array_equal(terminal.valid_mask, np.array([True, True, False]))
    assert np.array_equal(terminal.reference_valid_mask, terminal.valid_mask)
    assert not np.any(terminal.next_reference_valid_mask)
    assert np.allclose(terminal.action_chunk[-1], 0.0)
    assert np.allclose(terminal.ref_chunk[-1], 0.0)
    assert terminal.rewards[-1] == 0.0


def test_chunk_builder_never_crosses_step_gap() -> None:
    steps = _make_episode(length=8)
    for step in steps:
        step.done = False
    for step, step_id in zip(steps, (0, 1, 2, 3, 10, 11, 12, 13), strict=True):
        step.step_id = step_id

    transitions = build_chunk_transitions_from_episode(
        steps,
        chunk_len=3,
        stride=3,
        allow_partial=False,
    )

    assert len(transitions) == 2
    assert np.allclose(transitions[0].action_chunk[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(transitions[1].action_chunk[:, 0], np.array([5.0, 6.0, 7.0]))
    assert transitions[0].step_id == 0
    assert transitions[1].step_id == 10


def test_legacy_replay_mask_inference_is_fail_safe() -> None:
    terminal_full = build_terminal_aligned_chunk_transition(_make_episode(), chunk_len=3)
    assert terminal_full is not None
    full_record = terminal_full.to_journal_record()
    full_record.pop("valid_mask")
    full_record.pop("reference_valid_mask")
    full_record.pop("next_reference_valid_mask")
    restored = RLTTransition.from_mapping(full_record)
    assert np.all(restored.valid_mask)
    assert not np.any(restored.next_reference_valid_mask)

    terminal_partial = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)[-1]
    ambiguous_record = terminal_partial.to_journal_record()
    ambiguous_record.pop("valid_mask")
    ambiguous_record.pop("reference_valid_mask")
    ambiguous_record.pop("next_reference_valid_mask")
    with pytest.raises(ValueError, match="ambiguous zero-valued tail"):
        RLTTransition.from_mapping(ambiguous_record)


def test_legacy_nonterminal_next_reference_tail_requires_mask() -> None:
    transition = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)[1]
    assert not transition.done
    assert np.array_equal(transition.next_reference_valid_mask, np.array([True, False, False]))
    record = transition.to_journal_record()
    record.pop("next_reference_valid_mask")

    with pytest.raises(ValueError, match="ambiguous next-reference tail"):
        RLTTransition.from_mapping(record)


def test_replay_stores_collection_phase_id() -> None:
    transition = build_chunk_transitions_from_episode(
        _make_episode(collection_phase="warmup"),
        chunk_len=3,
        stride=2,
    )[0]
    record = transition.to_numpy()
    assert int(record["collection_phase_id"]) == COLLECTION_PHASE_WARMUP


def test_stratified_replay_samples_recent_warmup_and_human_pools() -> None:
    replay = ReplayBuffer(
        capacity=64,
        seed=0,
        sample_strategy="stratified",
        recent_episode_window=2,
        recent_online_ratio=0.4,
        warmup_demo_ratio=0.3,
        human_intervention_ratio=0.2,
    )
    transitions = []
    transitions.extend(
        build_chunk_transitions_from_episode(
            _make_episode(collection_phase="warmup", episode_id=1, policy_source=int(TransitionSource.BASE)),
            chunk_len=3,
            stride=3,
        )
    )
    transitions.extend(
        build_chunk_transitions_from_episode(
            _make_episode(collection_phase="online", episode_id=9, policy_source=int(TransitionSource.RL)),
            chunk_len=3,
            stride=3,
        )
    )
    transitions.extend(
        build_chunk_transitions_from_episode(
            _make_episode(collection_phase="online", episode_id=3, intervention_at=1),
            chunk_len=3,
            stride=3,
        )
    )
    replay.extend(transitions)

    batch = replay.sample(6)
    assert batch["z_rl"].shape[0] == 6
    assert np.any(batch["collection_phase_id"] == COLLECTION_PHASE_WARMUP)
    assert np.any((batch["collection_phase_id"] == COLLECTION_PHASE_ONLINE) & (batch["episode_id"] >= 8))
    assert np.any(batch["source_chunk"] == int(TransitionSource.HUMAN))


def test_replay_manager_journal_roundtrip(tmp_path) -> None:
    journal_path = tmp_path / "replay.pkl"
    manager = ReplayManager(32, journal_path=str(journal_path), seed=0)
    transitions = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)
    manager.add_transitions(transitions)
    assert manager.stats()["max_episode_id"] == 7
    restored = ReplayManager(32, journal_path=str(journal_path), seed=0)
    assert restored.stats()["size"] == len(transitions)
    assert restored.stats()["max_episode_id"] == 7


def test_replay_manager_rejects_journal_from_different_action_contract(tmp_path) -> None:
    journal_path = tmp_path / "replay.pkl"
    old_contract = ReplayTensorContract(z_dim=4, proprio_dim=3, chunk_len=3, action_dim=2)
    manager = ReplayManager(
        32,
        journal_path=str(journal_path),
        seed=0,
        tensor_contract=old_contract,
    )
    manager.add_transitions(build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2))

    current_contract = ReplayTensorContract(
        z_dim=4,
        proprio_dim=3,
        chunk_len=3,
        action_dim=3,
    )
    with pytest.raises(ValueError, match="replay journal.*ref_chunk has shape"):
        ReplayManager(
            32,
            journal_path=str(journal_path),
            seed=0,
            tensor_contract=current_contract,
        )


def test_replay_manager_rejects_new_transition_with_wrong_action_dim(tmp_path) -> None:
    contract = ReplayTensorContract(z_dim=4, proprio_dim=3, chunk_len=3, action_dim=2)
    manager = ReplayManager(
        32,
        journal_path=str(tmp_path / "replay.pkl"),
        seed=0,
        tensor_contract=contract,
    )
    transition = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)[0]
    transition.action_chunk = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="action_chunk has shape"):
        manager.add_transition(transition)


def test_nero_replay_contract_requires_finite_float32_19d_actions(tmp_path) -> None:
    steps = _make_episode()
    for step in steps:
        step.action = np.full((19,), step.step_id + 1, dtype=np.float16)
        step.ref_action = np.full((19,), step.step_id + 0.5, dtype=np.float16)
    transition = build_chunk_transitions_from_episode(steps, chunk_len=3, stride=3)[0]
    contract = ReplayTensorContract(z_dim=4, proprio_dim=3, chunk_len=3, action_dim=19)
    manager = ReplayManager(
        32,
        journal_path=str(tmp_path / "replay.pkl"),
        seed=0,
        tensor_contract=contract,
    )
    manager.add_transition(transition)
    batch = manager.sample_batch(1)
    assert batch["action_chunk"].shape == (1, 3, 19)
    assert batch["action_chunk"].dtype == np.float32
    assert batch["ref_chunk"].dtype == np.float32

    bad = transition.to_journal_record()
    bad["action_chunk"] = bad["action_chunk"].copy()
    bad["action_chunk"][0, 0] = np.nan
    with pytest.raises(ValueError, match="action_chunk contains non-finite"):
        RLTTransition.from_mapping(bad)


def test_replay_rejects_invalid_source_and_non_prefix_mask() -> None:
    transition = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)[0]
    record = transition.to_journal_record()
    record["source_chunk"] = np.array([0, 9, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="source_chunk contains values outside"):
        RLTTransition.from_mapping(record)

    record = transition.to_journal_record()
    record["valid_mask"] = np.array([True, False, True])
    record["reference_valid_mask"] = record["valid_mask"]
    with pytest.raises(ValueError, match="contiguous true prefix"):
        RLTTransition.from_mapping(record)


def test_replay_manager_batch_extend_fsyncs_once(tmp_path, monkeypatch) -> None:
    fsync_calls = []
    monkeypatch.setattr(replay_module.os, "fsync", lambda fd: fsync_calls.append(fd))

    manager = ReplayManager(32, journal_path=str(tmp_path / "replay.pkl"), seed=0)
    transitions = build_chunk_transitions_from_episode(_make_episode(), chunk_len=3, stride=2)
    manager.add_transitions(transitions)

    assert manager.stats()["size"] == len(transitions)
    assert len(fsync_calls) == 1
