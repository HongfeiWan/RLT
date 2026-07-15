"""Canonical Nero state and action contracts shared by offline and online bridges."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

ROT6D_CONVENTION: Final = "groot_row_major_first_two_rows"
ROT6D_COMPONENTS: Final = ("r00", "r01", "r02", "r10", "r11", "r12")

EEF_ACTION_CHANNEL_NAMES: Final = (
    "eef_9d.x",
    "eef_9d.y",
    "eef_9d.z",
    "eef_9d.rot6d.r00",
    "eef_9d.rot6d.r01",
    "eef_9d.rot6d.r02",
    "eef_9d.rot6d.r10",
    "eef_9d.rot6d.r11",
    "eef_9d.rot6d.r12",
)
HAND_ACTION_CHANNEL_NAMES: Final = (
    "hand_joint_target.thumb_cmc_pitch",
    "hand_joint_target.thumb_cmc_yaw",
    "hand_joint_target.index_mcp_pitch",
    "hand_joint_target.middle_mcp_pitch",
    "hand_joint_target.ring_mcp_pitch",
    "hand_joint_target.pinky_mcp_pitch",
    "hand_joint_target.index_mcp_roll",
    "hand_joint_target.ring_mcp_roll",
    "hand_joint_target.pinky_mcp_roll",
    "hand_joint_target.thumb_cmc_roll",
)
ARM_REFERENCE_CHANNEL_NAMES: Final = tuple(f"arm_joint_target.{index}" for index in range(7))

# Actor/critic proprioception deliberately excludes arm_joint_pos.  These names
# match the Machine-A wire layout emitted by GrootN1d7FeatureBackend.
ACTOR_PROPRIO_CHANNEL_NAMES: Final = (
    *(f"eef_9d[{index}]" for index in range(9)),
    *(f"hand_joint_pos[{index}]" for index in range(10)),
)

# Names written by the Teleop LeRobot v3 exporter. These describe the same
# physical row-first tensors as the runtime names below; the bridge validates
# them before assigning the runtime contract names.
V3_STATE_CHANNEL_NAMES: Final = (
    *(f"arm_joint_pos.{index}" for index in range(7)),
    "arm_eef_pos.x",
    "arm_eef_pos.y",
    "arm_eef_pos.z",
    "arm_eef_rot6d.r00",
    "arm_eef_rot6d.r01",
    "arm_eef_rot6d.r02",
    "arm_eef_rot6d.r10",
    "arm_eef_rot6d.r11",
    "arm_eef_rot6d.r12",
    *(
        f"hand_joint_pos.{name.removeprefix('hand_joint_target.')}"
        for name in HAND_ACTION_CHANNEL_NAMES
    ),
)
V3_ACTION_CHANNEL_NAMES: Final = (
    "arm_eef_pos_target.x",
    "arm_eef_pos_target.y",
    "arm_eef_pos_target.z",
    "arm_eef_rot6d_target.r00",
    "arm_eef_rot6d_target.r01",
    "arm_eef_rot6d_target.r02",
    "arm_eef_rot6d_target.r10",
    "arm_eef_rot6d_target.r11",
    "arm_eef_rot6d_target.r12",
    *HAND_ACTION_CHANNEL_NAMES,
)
V3_POLICY_SPACE_SCHEMA: Final = "nero_single_linker_l10_groot_v1_2_policy_space"

EXECUTED_ACTION_CHANNEL_NAMES: Final = EEF_ACTION_CHANNEL_NAMES + HAND_ACTION_CHANNEL_NAMES
VLA_REFERENCE_CHANNEL_NAMES: Final = EXECUTED_ACTION_CHANNEL_NAMES + ARM_REFERENCE_CHANNEL_NAMES
EXECUTED_ACTION_DIM: Final = len(EXECUTED_ACTION_CHANNEL_NAMES)
VLA_REFERENCE_DIM: Final = len(VLA_REFERENCE_CHANNEL_NAMES)
VLA_TO_EXECUTED_ACTION_INDICES: Final = tuple(range(EXECUTED_ACTION_DIM))

V3_STATE_DIM: Final = 26
V3_ACTION_DIM: Final = 19
V3_ARM_STATE_SLICE: Final = slice(0, 7)
V3_EEF_STATE_SLICE: Final = slice(7, 16)
V3_HAND_STATE_SLICE: Final = slice(16, 26)
ACTOR_PROPRIO_DIM: Final = len(ACTOR_PROPRIO_CHANNEL_NAMES)
V3_TO_ACTOR_PROPRIO_INDICES: Final = tuple(range(7, 26))


def semantic_layout_hash(
    channel_names: Sequence[str],
    *,
    rotation_convention: str | None,
) -> str:
    """Fingerprint channel order together with its rotation semantics."""

    material = {
        "channel_names": list(channel_names),
        "rotation_convention": rotation_convention,
    }
    payload = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_row_first_eef(eef_9d: np.ndarray, *, name: str) -> None:
    if eef_9d.shape[-1] != 9:
        raise ValueError(f"{name} must end in 9 EEF channels, got {eef_9d.shape}")
    if not np.isfinite(eef_9d).all():
        raise ValueError(f"{name} contains non-finite values")
    row0 = eef_9d[..., 3:6].astype(np.float64, copy=False)
    row1 = eef_9d[..., 6:9].astype(np.float64, copy=False)
    norm0 = np.linalg.norm(row0, axis=-1)
    orthogonal1 = row1 - np.sum(row0 * row1, axis=-1, keepdims=True) * row0 / np.maximum(
        np.square(norm0)[..., None], 1.0e-16
    )
    norm1 = np.linalg.norm(orthogonal1, axis=-1)
    if np.any(norm0 <= 1.0e-8) or np.any(norm1 <= 1.0e-8):
        raise ValueError(f"{name} contains degenerate row-first rot6d values")


def bridge_v3_policy_state_to_machine_a(
    state: Any,
    *,
    rotation_convention: str,
) -> np.ndarray:
    """Reorder canonical LeRobot v3 state into the validated inference order.

    The v3 state is ``arm7 + eef9 + hand10``. Machine A accepts
    ``eef9 + hand10 + arm7``. Both sides use the current inference rot6d order,
    so this bridge validates but does not transpose the rotation values.
    """

    if rotation_convention != ROT6D_CONVENTION:
        raise ValueError(
            f"v3 state rotation convention {rotation_convention!r} does not match "
            f"validated inference convention {ROT6D_CONVENTION!r}"
        )
    values = np.asarray(state, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != V3_STATE_DIM:
        raise ValueError(f"v3 state must have shape [..., {V3_STATE_DIM}], got {values.shape}")
    eef = values[..., V3_EEF_STATE_SLICE]
    _validate_row_first_eef(eef, name="v3 observation.state EEF")
    if not np.isfinite(values).all():
        raise ValueError("v3 observation.state contains non-finite values")
    return np.concatenate(
        (
            eef,
            values[..., V3_HAND_STATE_SLICE],
            values[..., V3_ARM_STATE_SLICE],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def project_v3_policy_state_to_actor_proprio(
    state: Any,
    *,
    rotation_convention: str,
) -> np.ndarray:
    """Project the 26D v3 state onto actor/critic EEF-and-hand proprioception.

    The complete ``arm7 + eef9 + hand10`` state remains the input to the frozen
    400k VLA.  Only ``eef9 + hand10`` crosses the replay/actor/critic boundary.
    """

    # Reuse the full checkpoint-input bridge so rotation and finiteness checks
    # cannot drift between the VLA and actor/critic paths.
    checkpoint_state = bridge_v3_policy_state_to_machine_a(
        state,
        rotation_convention=rotation_convention,
    )
    projected = checkpoint_state[..., :ACTOR_PROPRIO_DIM]
    if projected.shape[-1] != ACTOR_PROPRIO_DIM:  # pragma: no cover - invariant guard.
        raise RuntimeError("actor proprio projection produced an invalid dimension")
    return projected.astype(np.float32, copy=True)


def bridge_v3_executed_action(
    action: Any,
    *,
    rotation_convention: str,
) -> np.ndarray:
    """Validate and copy the real 19D action exported by the HIL bridge."""

    if rotation_convention != ROT6D_CONVENTION:
        raise ValueError(
            f"v3 action rotation convention {rotation_convention!r} does not match "
            f"validated inference convention {ROT6D_CONVENTION!r}"
        )
    values = np.asarray(action, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != V3_ACTION_DIM:
        raise ValueError(f"v3 action must have shape [..., {V3_ACTION_DIM}], got {values.shape}")
    _validate_row_first_eef(values[..., :9], name="v3 action EEF")
    if not np.isfinite(values).all():
        raise ValueError("v3 action contains non-finite values")
    return values.copy()


def project_vla_reference_to_executed_action(reference: Any) -> np.ndarray:
    """Project a complete 26D VLA reference onto the real 19D command space."""

    values = np.asarray(reference, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != VLA_REFERENCE_DIM:
        raise ValueError(
            f"VLA reference must have shape [..., {VLA_REFERENCE_DIM}], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("VLA reference contains non-finite values")
    projected = values[..., VLA_TO_EXECUTED_ACTION_INDICES]
    return projected.astype(np.float32, copy=True)
