from __future__ import annotations

import numpy as np
import pytest

from groot_rlt.integration.nero_action_contract import (
    ACTOR_PROPRIO_DIM,
    EXECUTED_ACTION_CHANNEL_NAMES,
    ROT6D_CONVENTION,
    VLA_REFERENCE_CHANNEL_NAMES,
    bridge_v3_executed_action,
    bridge_v3_policy_state_to_machine_a,
    project_v3_policy_state_to_actor_proprio,
    project_vla_reference_to_executed_action,
    semantic_layout_hash,
)


def _row_first_eef() -> np.ndarray:
    return np.asarray((0.1, 0.2, 0.3, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32)


def test_v3_state_bridge_reorders_without_changing_inference_rot6d() -> None:
    arm = np.arange(7, dtype=np.float32) + 100.0
    eef = _row_first_eef()
    hand = np.arange(10, dtype=np.float32) + 200.0
    v3_state = np.concatenate((arm, eef, hand))

    bridged = bridge_v3_policy_state_to_machine_a(
        v3_state,
        rotation_convention=ROT6D_CONVENTION,
    )

    np.testing.assert_array_equal(bridged[:9], eef)
    np.testing.assert_array_equal(bridged[9:19], hand)
    np.testing.assert_array_equal(bridged[19:26], arm)
    np.testing.assert_array_equal(bridged[3:9], v3_state[10:16])

    actor_proprio = project_v3_policy_state_to_actor_proprio(
        v3_state,
        rotation_convention=ROT6D_CONVENTION,
    )
    assert actor_proprio.shape == (ACTOR_PROPRIO_DIM,)
    np.testing.assert_array_equal(actor_proprio[:9], eef)
    np.testing.assert_array_equal(actor_proprio[9:], hand)
    assert not np.isin(arm, actor_proprio).all()


def test_v3_bridge_rejects_unvalidated_rotation_convention() -> None:
    state = np.concatenate((np.zeros(7), _row_first_eef(), np.zeros(10))).astype(np.float32)
    with pytest.raises(ValueError, match="does not match validated inference convention"):
        bridge_v3_policy_state_to_machine_a(
            state,
            rotation_convention="mismatched_rot6d_convention",
        )


def test_v3_action_is_the_real_19d_command() -> None:
    action = np.concatenate((_row_first_eef(), np.arange(10, dtype=np.float32)))

    bridged = bridge_v3_executed_action(action, rotation_convention=ROT6D_CONVENTION)

    assert bridged.shape == (19,)
    np.testing.assert_array_equal(bridged, action)
    assert not np.shares_memory(bridged, action)


def test_vla_projection_excludes_arm_tail_exactly() -> None:
    reference = np.arange(2 * 26, dtype=np.float32).reshape(2, 26)
    changed_tail = reference.copy()
    changed_tail[:, 19:] += 10000.0

    projected = project_vla_reference_to_executed_action(reference)
    projected_changed_tail = project_vla_reference_to_executed_action(changed_tail)

    assert projected.shape == (2, 19)
    np.testing.assert_array_equal(projected, reference[:, :19])
    np.testing.assert_array_equal(projected_changed_tail, projected)


def test_layout_hash_includes_rot6d_convention_and_projection() -> None:
    source_hash = semantic_layout_hash(
        VLA_REFERENCE_CHANNEL_NAMES,
        rotation_convention=ROT6D_CONVENTION,
    )
    target_hash = semantic_layout_hash(
        EXECUTED_ACTION_CHANNEL_NAMES,
        rotation_convention=ROT6D_CONVENTION,
    )
    mismatch_hash = semantic_layout_hash(
        EXECUTED_ACTION_CHANNEL_NAMES,
        rotation_convention="mismatched_rot6d_convention",
    )

    assert source_hash != target_hash
    assert target_hash != mismatch_hash
