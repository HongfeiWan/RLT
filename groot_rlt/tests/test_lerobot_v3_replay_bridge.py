# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from groot_rlt.integration.lerobot_v3_replay_bridge import (
    ReplayFrameFeatures,
    ReplaySource,
    build_lerobot_v3_replay_bundle,
    open_official_lerobot_dataset,
    write_replay_bundle,
)
from groot_rlt.integration.nero_action_contract import (
    ROT6D_CONVENTION,
    V3_ACTION_CHANNEL_NAMES,
    V3_POLICY_SPACE_SCHEMA,
    V3_STATE_CHANNEL_NAMES,
)

_FEATURE_FINGERPRINT = "sha256:" + "1" * 64


def _feature(dtype: str, shape: list[int], names: list[str] | None = None) -> dict:
    return {"dtype": dtype, "shape": shape, "names": names}


def _info(*, total_frames: int, total_episodes: int = 1) -> dict:
    return {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": {
            "observation.state": _feature("float32", [26], list(V3_STATE_CHANNEL_NAMES)),
            "action": _feature("float32", [19], list(V3_ACTION_CHANNEL_NAMES)),
            "intervention": _feature("bool", [1]),
            "teleop_stack.is_intervention": _feature("bool", [1]),
            "teleop_stack.task_episode_id": _feature("string", [1]),
            "teleop_stack.partition": _feature("string", [1]),
            "teleop_stack.behavior_source": _feature("string", [1]),
            "teleop_stack.advantage_label_rule": _feature("string", [1]),
            "teleop_stack.terminal_label": _feature("string", [1]),
            "next.reward": _feature("float32", [1]),
            "next.done": _feature("bool", [1]),
        },
    }


def _sidecar_episode(index: int, *, length: int, label: str) -> dict:
    return {
        "episode_index": index,
        "task_episode_id": f"task_episode_{index:03d}",
        "task": "pick and place",
        "terminal_label": label,
        "success": label == "success",
        "length": length,
        "source_episode_dir": f"episode_{index:03d}",
    }


def _recap(*, episode_lengths: list[int], labels: list[str] | None = None) -> dict:
    labels = labels or ["success"] * len(episode_lengths)
    return {
        "schema_version": "teleop_stack.lerobot_v3_dagger.v1",
        "format_name": "lerobot_v3_dagger",
        "repo_id": "local/test",
        "raw_capture_id": "capture_test",
        "normalization_schema": V3_POLICY_SPACE_SCHEMA,
        "fps": 10,
        "excluded_partitions": ["stop_hold", "outside_episode"],
        "policy_space": {
            "observation_eef_frame": "policy_state",
            "model_eef_frame": "policy_state",
            "command_eef_frame": "genesis_world",
            "state_to_genesis_transform": {
                "source_frame": "policy_state",
                "target_frame": "genesis_world",
                "rot6d_convention": ROT6D_CONVENTION,
                "translation_xyz": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "eef_offset_translation_xyz": [0.0, 0.0, 0.0],
                "eef_offset_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "episodes": [
            _sidecar_episode(index, length=length, label=labels[index])
            for index, length in enumerate(episode_lengths)
        ],
    }


def _state(frame_index: int) -> np.ndarray:
    arm = np.arange(7, dtype=np.float32) + 100.0 + frame_index
    eef = np.asarray(
        [0.1 + frame_index, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    hand = np.arange(10, dtype=np.float32) + 200.0 + frame_index
    return np.concatenate((arm, eef, hand))


def _action(frame_index: int) -> np.ndarray:
    eef = np.asarray(
        [0.4 + frame_index, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    hand = np.arange(10, dtype=np.float32) + 10.0 + frame_index
    return np.concatenate((eef, hand))


_PARTITION_VALUES = {
    "policy_rollout": ("policy", "pending_value_function", False),
    "human_correction": ("human", "forced_positive", True),
    "human_demo": ("human", "positive_demo", False),
}


def _rows(
    partitions: list[str],
    *,
    episode_index: int = 0,
    label: str = "success",
) -> list[dict]:
    result = []
    for frame_index, partition in enumerate(partitions):
        behavior, advantage, intervention = _PARTITION_VALUES[partition]
        done = frame_index == len(partitions) - 1
        result.append(
            {
                "episode_index": np.asarray(episode_index, dtype=np.int64),
                "frame_index": np.asarray(frame_index, dtype=np.int64),
                "timestamp": np.asarray(frame_index / 10.0, dtype=np.float32),
                "observation.state": _state(frame_index),
                "action": _action(frame_index),
                "intervention": np.asarray([intervention], dtype=np.bool_),
                "teleop_stack.is_intervention": np.asarray([intervention], dtype=np.bool_),
                "teleop_stack.task_episode_id": np.asarray([f"task_episode_{episode_index:03d}"]),
                "teleop_stack.partition": np.asarray([partition]),
                "teleop_stack.behavior_source": np.asarray([behavior]),
                "teleop_stack.advantage_label_rule": np.asarray([advantage]),
                "teleop_stack.terminal_label": np.asarray([label]),
                "next.reward": np.asarray(
                    [1.0 if done and label == "success" else 0.0], dtype=np.float32
                ),
                "next.done": np.asarray([done], dtype=np.bool_),
            }
        )
    return result


class _DuckLoader:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.requested_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        self.requested_indices.append(index)
        return self.rows[index]


class _FeatureProvider:
    def __init__(self):
        self.identities = []

    def __call__(self, identity, row):
        self.identities.append(identity)
        frame_index = int(np.asarray(row["frame_index"]).item())
        reference = np.concatenate((_action(frame_index), np.arange(7, dtype=np.float32) + 500.0))
        return ReplayFrameFeatures(
            z_rl=np.arange(4, dtype=np.float32) + frame_index,
            vla_reference_action=reference,
        )


class LeRobotV3ReplayBridgeTest(unittest.TestCase):
    def test_bridge_splits_partitions_and_preserves_real_action_contract(self) -> None:
        partitions = [
            "policy_rollout",
            "policy_rollout",
            "human_correction",
            "human_correction",
            "human_demo",
            "human_demo",
        ]
        rows = _rows(partitions)
        loader = _DuckLoader(rows)
        provider = _FeatureProvider()

        bundle = build_lerobot_v3_replay_bundle(
            loader,
            info_payload=_info(total_frames=len(rows)),
            recap_payload=_recap(episode_lengths=[len(rows)]),
            feature_provider=provider,
            feature_contract_fingerprint=_FEATURE_FINGERPRINT,
        )

        self.assertEqual(loader.requested_indices, list(range(len(rows))))
        self.assertEqual(len(provider.identities), len(rows))
        self.assertTrue(
            all(i.dataset_fingerprint == bundle.dataset_fingerprint for i in provider.identities)
        )
        self.assertEqual([segment.partition for segment in bundle.segments], partitions[::2])
        self.assertEqual(bundle.dropped_boundary_frame_count, 2)
        self.assertEqual(
            [
                [(step.source_frame_index, step.next_source_frame_index) for step in segment.steps]
                for segment in bundle.segments
            ],
            [[(0, 1)], [(2, 3)], [(4, 5), (5, 5)]],
        )

        policy, correction, demo = bundle.segments
        self.assertEqual(policy.steps[0].source, int(ReplaySource.BASE))
        self.assertFalse(policy.steps[0].intervention_flag)
        self.assertEqual(correction.steps[0].source, int(ReplaySource.HUMAN))
        self.assertTrue(correction.steps[0].intervention_flag)
        self.assertEqual(demo.steps[0].source, int(ReplaySource.HUMAN))
        self.assertFalse(demo.steps[0].intervention_flag)

        np.testing.assert_array_equal(correction.steps[0].action, rows[2]["action"])
        np.testing.assert_array_equal(correction.steps[0].ref_action, _action(2))
        source_state = rows[2]["observation.state"]
        expected_proprio = np.concatenate(
            (source_state[7:16], source_state[16:26], source_state[0:7])
        )
        np.testing.assert_array_equal(correction.steps[0].proprio, expected_proprio)

        terminal = demo.steps[-1]
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 1.0)
        self.assertEqual(terminal.success, 1)
        np.testing.assert_array_equal(terminal.z_rl, terminal.next_z_rl)
        self.assertEqual(bundle.outcome_counts, {"success": 1, "failure": 0})

        payload = bundle.to_payload()
        self.assertEqual(len(payload["episodes"]), 3)
        self.assertEqual(
            payload["manifest"]["partition_boundary_policy"], "split_and_drop_nonterminal_tail"
        )
        mapped_step = payload["episodes"][1]["steps"][0]
        self.assertEqual(mapped_step["source_partition"], "human_correction")
        self.assertEqual(mapped_step["source_frame_index"], 2)
        self.assertEqual(mapped_step["dataset_fingerprint"], bundle.dataset_fingerprint)
        self.assertEqual(mapped_step["segment_id"], payload["episodes"][1]["segment_id"])

    def test_failure_episode_is_explicit_terminal_with_zero_reward(self) -> None:
        rows = _rows(["policy_rollout", "policy_rollout"], label="failure")
        bundle = build_lerobot_v3_replay_bundle(
            rows,
            info_payload=_info(total_frames=2),
            recap_payload=_recap(episode_lengths=[2], labels=["failure"]),
            feature_provider=_FeatureProvider(),
            feature_contract_fingerprint=_FEATURE_FINGERPRINT,
        )

        terminal = bundle.segments[0].steps[-1]
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 0.0)
        self.assertEqual(terminal.success, 0)
        self.assertEqual(bundle.outcome_counts, {"success": 0, "failure": 1})

    def test_intervention_must_match_both_mirror_and_partition(self) -> None:
        base_rows = _rows(["policy_rollout", "human_correction"])
        mutations = (
            lambda rows: rows[0].__setitem__("intervention", np.asarray([True], dtype=np.bool_)),
            lambda rows: rows[1].__setitem__(
                "teleop_stack.is_intervention", np.asarray([False], dtype=np.bool_)
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(base_rows)
                mutation(rows)
                with self.assertRaisesRegex(ValueError, "intervention mismatch"):
                    build_lerobot_v3_replay_bundle(
                        rows,
                        info_payload=_info(total_frames=2),
                        recap_payload=_recap(episode_lengths=[2]),
                        feature_provider=_FeatureProvider(),
                        feature_contract_fingerprint=_FEATURE_FINGERPRINT,
                    )

    def test_bridge_rejects_unknown_partition_and_untrainable_terminal(self) -> None:
        rows = _rows(["policy_rollout", "policy_rollout"])
        rows[0]["teleop_stack.partition"] = np.asarray(["stop_hold"])
        with self.assertRaisesRegex(ValueError, "non-training partition"):
            build_lerobot_v3_replay_bundle(
                rows,
                info_payload=_info(total_frames=2),
                recap_payload=_recap(episode_lengths=[2]),
                feature_provider=_FeatureProvider(),
                feature_contract_fingerprint=_FEATURE_FINGERPRINT,
            )

        rows = _rows(["policy_rollout", "policy_rollout"])
        for row in rows:
            row["teleop_stack.terminal_label"] = np.asarray(["aborted"])
        recap = _recap(episode_lengths=[2])
        recap["episodes"][0]["terminal_label"] = "aborted"
        recap["episodes"][0]["success"] = False
        with self.assertRaisesRegex(ValueError, "terminal_label must be success or failure"):
            build_lerobot_v3_replay_bundle(
                rows,
                info_payload=_info(total_frames=2),
                recap_payload=recap,
                feature_provider=_FeatureProvider(),
                feature_contract_fingerprint=_FEATURE_FINGERPRINT,
            )

    def test_bridge_rejects_wrong_terminal_reward_and_rot6d_metadata(self) -> None:
        rows = _rows(["policy_rollout", "policy_rollout"])
        rows[-1]["next.reward"] = np.asarray([0.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "next.reward=1.0"):
            build_lerobot_v3_replay_bundle(
                rows,
                info_payload=_info(total_frames=2),
                recap_payload=_recap(episode_lengths=[2]),
                feature_provider=_FeatureProvider(),
                feature_contract_fingerprint=_FEATURE_FINGERPRINT,
            )

        info = _info(total_frames=2)
        names = info["features"]["observation.state"]["names"]
        names[10], names[13] = names[13], names[10]
        with self.assertRaisesRegex(
            ValueError, "validated groot_row_major_first_two_rows contract"
        ):
            build_lerobot_v3_replay_bundle(
                _rows(["policy_rollout", "policy_rollout"]),
                info_payload=info,
                recap_payload=_recap(episode_lengths=[2]),
                feature_provider=_FeatureProvider(),
                feature_contract_fingerprint=_FEATURE_FINGERPRINT,
            )

    def test_bridge_checks_feature_shape_and_expected_fingerprint_before_features(self) -> None:
        rows = _rows(["policy_rollout", "policy_rollout"])

        def wrong_reference(identity, row):
            del identity, row
            return {"z_rl": np.zeros(4), "vla_reference_action": np.zeros(19)}

        with self.assertRaisesRegex(ValueError, "feature generation failed"):
            build_lerobot_v3_replay_bundle(
                rows,
                info_payload=_info(total_frames=2),
                recap_payload=_recap(episode_lengths=[2]),
                feature_provider=wrong_reference,
                feature_contract_fingerprint=_FEATURE_FINGERPRINT,
            )

        provider = _FeatureProvider()
        with self.assertRaisesRegex(ValueError, "dataset fingerprint mismatch"):
            build_lerobot_v3_replay_bundle(
                rows,
                info_payload=_info(total_frames=2),
                recap_payload=_recap(episode_lengths=[2]),
                feature_provider=provider,
                feature_contract_fingerprint=_FEATURE_FINGERPRINT,
                expected_dataset_fingerprint="sha256:" + "f" * 64,
            )
        self.assertEqual(provider.identities, [])

    def test_duck_factory_and_atomic_writer(self) -> None:
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return "loader"

        self.assertEqual(
            open_official_lerobot_dataset("repo", root="dataset", dataset_factory=factory),
            "loader",
        )
        self.assertEqual(calls, [(("repo",), {"root": "dataset"})])

        rows = _rows(["policy_rollout", "policy_rollout"])
        bundle = build_lerobot_v3_replay_bundle(
            rows,
            info_payload=_info(total_frames=2),
            recap_payload=_recap(episode_lengths=[2]),
            feature_provider=_FeatureProvider(),
            feature_contract_fingerprint=_FEATURE_FINGERPRINT,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay" / "input.pkl"
            self.assertEqual(write_replay_bundle(bundle, path), path.resolve())
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            self.assertEqual(payload["manifest"]["bridge_fingerprint"], bundle.bridge_fingerprint)
            with self.assertRaises(FileExistsError):
                write_replay_bundle(bundle, path)


if __name__ == "__main__":
    unittest.main()
