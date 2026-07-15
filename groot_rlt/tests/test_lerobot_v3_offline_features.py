# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from groot_rlt.integration.lerobot_v3_offline_features import (
    AtomicFeatureCache,
    CameraBinding,
    GrootOfflineMachineABackend,
    LocalV3ParquetPyAVLoader,
    OfflineFeatureContract,
    OfflineInferenceResult,
    OfflineV3MachineAFeatureProvider,
    camera_contract_fingerprint,
    dataset_content_fingerprint,
    frame_seed,
    inspect_deployment_contract,
    open_v3_dataset_loader,
    token_contract_fingerprint,
    write_offline_bridge_bundle,
)
from groot_rlt.integration.lerobot_v3_replay_bridge import (
    FrameIdentity,
    ReplayBridgeBundle,
    ReplayBridgeSegment,
    ReplayBridgeStep,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _source_state() -> np.ndarray:
    arm = np.arange(7, dtype=np.float32) + 10.0
    eef = np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    hand = np.arange(10, dtype=np.float32) + 20.0
    return np.concatenate((arm, eef, hand))


def _reference_action() -> np.ndarray:
    eef = np.asarray([0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    hand = np.arange(10, dtype=np.float32) + 30.0
    arm = np.arange(7, dtype=np.float32) + 40.0
    return np.concatenate((eef, hand, arm))


def _bindings() -> tuple[CameraBinding, ...]:
    return (
        CameraBinding("observation.images.ego_view", "ego_view", 2, 3),
        CameraBinding("observation.images.wrist_view", "wrist_view", 2, 3),
    )


def _contract() -> OfflineFeatureContract:
    bindings = _bindings()
    return OfflineFeatureContract(
        dataset_fingerprint=_sha("a"),
        dataset_content_fingerprint=_sha("0"),
        checkpoint_fingerprint=_sha("b"),
        encoder_artifact_fingerprint=_sha("c"),
        processor_fingerprint=_sha("d"),
        prefix_cache_fingerprint=_sha("e"),
        vlm_deployment_content_fingerprint=_sha("f"),
        token_contract_fingerprint=token_contract_fingerprint(
            token_scope="image", token_sampling="uniform", max_vl_tokens=3
        ),
        camera_contract_fingerprint=camera_contract_fingerprint(bindings),
        model_path="/deployment/model",
        processor_path="/deployment/model",
        vlm_model_path="/deployment/vlm",
        prefix_cache_manifest_path="/deployment/cache/manifest.json",
        token_scope="image",
        token_sampling="uniform",
        max_vl_tokens=3,
        denoise_steps=7,
        base_seed=123,
        z_dim=4,
        camera_bindings=bindings,
    )


def _identity(contract: OfflineFeatureContract) -> FrameIdentity:
    return FrameIdentity(
        dataset_fingerprint=contract.dataset_fingerprint,
        episode_index=0,
        frame_index=0,
        task_episode_id="task_episode_000",
        partition="policy_rollout",
    )


def _row(*, images: tuple[Any, Any] | None = None) -> dict[str, Any]:
    if images is None:
        ego = np.linspace(0.0, 1.0, 18, dtype=np.float32).reshape(3, 2, 3)
        wrist = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        images = (ego, wrist)
    return {
        "episode_index": 0,
        "frame_index": 0,
        "observation.state": _source_state(),
        "teleop_stack.task_episode_id": "task_episode_000",
        "teleop_stack.partition": "policy_rollout",
        "task": "pick and place",
        "observation.images.ego_view": images[0],
        "observation.images.wrist_view": images[1],
    }


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], int]] = []

    def infer_one(self, observation: dict[str, Any], *, seed: int) -> OfflineInferenceResult:
        self.calls.append((observation, seed))
        state = observation["state"]
        proprio = np.concatenate(
            (state["eef_9d"][0, -1], state["hand_joint_pos"][0, -1])
        )
        return OfflineInferenceResult(
            z_rl=np.arange(4, dtype=np.float32),
            vla_reference_action=_reference_action(),
            proprio=proprio,
        )


class _NeverDecode:
    def numpy(self) -> np.ndarray:
        raise AssertionError("a cache hit must not decode video")


class OfflineFeatureProviderTest(unittest.TestCase):
    def test_video_byte_change_invalidates_content_fingerprint_and_cache_contract(self) -> None:
        camera_keys = tuple(binding.source_key for binding in _bindings())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (dataset / "data" / "chunk-000").mkdir(parents=True)
            for camera_key in camera_keys:
                (dataset / "videos" / camera_key / "chunk-000").mkdir(parents=True)
            (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")
            (dataset / "meta" / "teleop_stack_recap.json").write_text(
                "{}", encoding="utf-8"
            )
            (dataset / "meta" / "tasks.parquet").write_bytes(b"tasks")
            (dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(
                b"episodes"
            )
            (dataset / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"rows")
            video_paths = []
            for index, camera_key in enumerate(camera_keys):
                path = dataset / "videos" / camera_key / "chunk-000" / "file-000.mp4"
                path.write_bytes(b"fake-mp4-" + bytes([index]))
                video_paths.append(path)

            first = dataset_content_fingerprint(dataset, camera_keys=camera_keys)
            first_contract = dataclasses.replace(
                _contract(), dataset_content_fingerprint=first
            )
            cache_dir = root / "cache"
            AtomicFeatureCache(cache_dir, first_contract)

            video_bytes = bytearray(video_paths[0].read_bytes())
            video_bytes[-1] ^= 1
            video_paths[0].write_bytes(video_bytes)
            second = dataset_content_fingerprint(dataset, camera_keys=camera_keys)
            second_contract = dataclasses.replace(
                first_contract, dataset_content_fingerprint=second
            )

            self.assertNotEqual(first, second)
            self.assertNotEqual(
                first_contract.feature_contract_fingerprint,
                second_contract.feature_contract_fingerprint,
            )
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                AtomicFeatureCache(cache_dir, second_contract)

    def test_observation_contract_and_resumable_atomic_cache(self) -> None:
        contract = _contract()
        identity = _identity(contract)
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as temporary:
            cache = AtomicFeatureCache(temporary, contract)
            provider = OfflineV3MachineAFeatureProvider(backend, contract, cache)
            first = provider(identity, _row())

            self.assertEqual(provider.cache_misses, 1)
            self.assertEqual(provider.cache_hits, 0)
            self.assertEqual(len(backend.calls), 1)
            observation, seed = backend.calls[0]
            self.assertEqual(seed, frame_seed(identity, base_seed=contract.base_seed))
            np.testing.assert_array_equal(
                observation["state"]["arm_joint_pos"][0, 0], _source_state()[:7]
            )
            np.testing.assert_array_equal(
                observation["state"]["eef_9d"][0, 0], _source_state()[7:16]
            )
            np.testing.assert_array_equal(
                observation["state"]["hand_joint_pos"][0, 0], _source_state()[16:]
            )
            self.assertEqual(observation["video"]["ego_view"].shape, (1, 1, 2, 3, 3))
            self.assertEqual(observation["video"]["ego_view"].dtype, np.uint8)
            self.assertEqual(observation["language"][contract.language_key], [["pick and place"]])

            no_decode = (_NeverDecode(), _NeverDecode())
            resumed_backend = _FakeBackend()
            resumed = OfflineV3MachineAFeatureProvider(
                resumed_backend, contract, AtomicFeatureCache(temporary, contract)
            )
            second = resumed(identity, _row(images=no_decode))
            np.testing.assert_array_equal(second.z_rl, first.z_rl)
            np.testing.assert_array_equal(second.vla_reference_action, first.vla_reference_action)
            self.assertEqual(resumed.cache_hits, 1)
            self.assertEqual(resumed.cache_misses, 0)
            self.assertEqual(resumed_backend.calls, [])
            self.assertEqual(cache.summary()["temporary_file_count"], 0)

    def test_cache_and_frame_identity_mismatches_fail_closed(self) -> None:
        contract = _contract()
        identity = _identity(contract)
        with tempfile.TemporaryDirectory() as temporary:
            cache = AtomicFeatureCache(temporary, contract)
            provider = OfflineV3MachineAFeatureProvider(_FakeBackend(), contract, cache)
            provider(identity, _row())

            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                AtomicFeatureCache(
                    temporary, dataclasses.replace(contract, base_seed=contract.base_seed + 1)
                )

            cache.frame_path(identity).write_bytes(b"not an npz")
            with self.assertRaisesRegex(ValueError, "invalid cached feature frame"):
                provider(identity, _row())

        wrong_identity = dataclasses.replace(identity, frame_index=1)
        with tempfile.TemporaryDirectory() as temporary:
            provider = OfflineV3MachineAFeatureProvider(
                _FakeBackend(), contract, AtomicFeatureCache(temporary, contract)
            )
            with self.assertRaisesRegex(ValueError, "row episode/frame"):
                provider(wrong_identity, _row())

    def test_bridge_pickle_is_atomic_round_trip_and_refuses_overwrite(self) -> None:
        step = ReplayBridgeStep(
            z_rl=np.arange(4, dtype=np.float32),
            proprio=np.arange(19, dtype=np.float32),
            vla_reference_action=np.arange(26, dtype=np.float32),
            ref_action=np.arange(19, dtype=np.float32),
            action=np.arange(19, dtype=np.float32),
            reward=1.0,
            done=True,
            next_z_rl=np.arange(4, dtype=np.float32),
            next_proprio=np.arange(19, dtype=np.float32),
            source=0,
            collection_phase="warmup",
            success=1,
            intervention_flag=False,
            episode_id=0,
            step_id=0,
            source_frame_index=0,
            next_source_frame_index=0,
        )
        segment = ReplayBridgeSegment(
            segment_id="dataset:episode_000000:run_0000:policy_rollout:frames_000000_000000",
            episode_index=0,
            task_episode_id="task_episode_000",
            partition="policy_rollout",
            source_frame_start=0,
            source_frame_end_inclusive=0,
            dropped_tail_frame_index=None,
            steps=(step,),
        )
        bundle = ReplayBridgeBundle(
            dataset_fingerprint=_sha("a"),
            feature_contract_fingerprint=_sha("b"),
            bridge_fingerprint=_sha("c"),
            fps=10.0,
            source_frame_count=1,
            replay_step_count=1,
            dropped_boundary_frame_count=0,
            outcome_counts={"success": 1, "failure": 0},
            segments=(segment,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bridge.pkl"
            artifact = write_offline_bridge_bundle(bundle, output)
            self.assertEqual(artifact["path"], str(output.resolve()))
            self.assertEqual(artifact["size_bytes"], output.stat().st_size)
            self.assertEqual(
                artifact["sha256"],
                "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            with output.open("rb") as stream:
                payload = pickle.load(stream)
            self.assertEqual(payload["schema_name"], "groot_rlt.lerobot_v3_dagger_replay")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["manifest"], bundle.manifest())
            self.assertEqual(len(payload["episodes"]), 1)
            self.assertEqual(len(payload["episodes"][0]["steps"]), 1)
            self.assertEqual(payload["episodes"][0]["steps"][0]["action"].shape, (19,))
            self.assertEqual(
                payload["episodes"][0]["steps"][0]["vla_reference_action"].shape,
                (26,),
            )
            self.assertFalse(list(output.parent.glob("*.tmp")))
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_offline_bridge_bundle(bundle, output)


class _FakeHead:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.num_inference_timesteps = 1
        self.encode_calls = 0
        self.action_calls = 0

    def _encode_features(self, backbone_output: dict[str, Any], action_inputs: Any) -> Any:
        del action_inputs
        self.encode_calls += 1
        backbone_output["raw"].fill_(-1000.0)
        return types.SimpleNamespace(
            backbone_features=self.torch.ones((1, 1, 2)),
            state_features=self.torch.ones((1, 1, 2)),
        )

    def get_action_with_features(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.action_calls += 1
        return {"action_pred": self.torch.rand((1, 32, 26))}


class _FakeModel:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.config = types.SimpleNamespace(backbone_embedding_dim=8)
        self.action_head = _FakeHead(torch_module)
        self.backbone_calls = 0

    def prepare_input(self, **collated: Any) -> tuple[dict[str, Any], Any]:
        del collated
        return {}, types.SimpleNamespace(embodiment_id=self.torch.tensor([0]))

    def backbone(self, inputs: dict[str, Any]) -> dict[str, Any]:
        del inputs
        self.backbone_calls += 1
        return {"raw": self.torch.arange(24, dtype=self.torch.float32).reshape(1, 3, 8)}


class _FakeProcessor:
    def decode_action(
        self, normalized: np.ndarray, embodiment_tag: str, states: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        del embodiment_tag
        self.states = states
        return {
            "eef_9d": normalized[..., :9],
            "hand_joint_target": normalized[..., 9:19],
            "arm_joint_target": normalized[..., 19:26],
        }


class _FakePolicy:
    def __init__(self, torch_module: Any) -> None:
        config = lambda keys, delta=None: types.SimpleNamespace(  # noqa: E731
            modality_keys=list(keys), delta_indices=[0] if delta is None else delta
        )
        self.modality_configs = {
            "video": config(("ego_view", "wrist_view")),
            "state": config(("eef_9d", "hand_joint_pos", "arm_joint_pos")),
            "action": config(("eef_9d", "hand_joint_target", "arm_joint_target"), list(range(32))),
            "language": config(("annotation.human.action.task_description",)),
        }
        self.language_key = "annotation.human.action.task_description"
        self.embodiment_tag = "new_embodiment"
        self.strict = True
        self.model = _FakeModel(torch_module)
        self.processor = _FakeProcessor()
        self.checked = 0

    def check_observation(self, observation: dict[str, Any]) -> None:
        del observation
        self.checked += 1

    def _unbatch_observation(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "video": {key: value[0] for key, value in observation["video"].items()},
                "state": {key: value[0] for key, value in observation["state"].items()},
                "language": {key: value[0] for key, value in observation["language"].items()},
            }
        ]

    def collate_fn(self, processed: list[Any]) -> dict[str, Any]:
        self.processed = processed
        return {}

    def _rec_to_dtype(self, value: dict[str, Any], *, dtype: Any) -> dict[str, Any]:
        del dtype
        return value


class _FakeEncoder:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.config = types.SimpleNamespace(rl_token_dim=4, max_vl_tokens=3, input_dim=8)

    def encode_rl_token(self, packed: Any, mask: Any) -> Any:
        self.packed = packed.clone()
        self.mask = mask.clone()
        return packed[:, 0, :4]


class GrootOfflineBackendTest(unittest.TestCase):
    def test_fake_policy_cpu_is_one_pass_and_seed_deterministic(self) -> None:
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        contract = _contract()
        policy = _FakePolicy(torch)
        encoder = _FakeEncoder(torch)

        def pack(backbone_output: dict[str, Any], **kwargs: Any) -> tuple[Any, ...]:
            self.assertEqual(
                kwargs,
                {"token_scope": "image", "max_tokens": 3, "token_sampling": "uniform"},
            )
            raw = backbone_output["raw"].clone()
            return raw, torch.ones((1, 3), dtype=torch.bool), None, [3], [3]

        backend = GrootOfflineMachineABackend(
            policy=policy,
            encoder=encoder,
            pack_vl_tokens=pack,
            torch_module=torch,
            device="cpu",
            contract=contract,
            processor_message_factory=lambda _policy, item: item,
        )
        state = _source_state()
        observation = {
            "video": {
                "ego_view": np.zeros((1, 1, 2, 3, 3), dtype=np.uint8),
                "wrist_view": np.zeros((1, 1, 2, 3, 3), dtype=np.uint8),
            },
            "state": {
                "eef_9d": state[7:16][None, None],
                "hand_joint_pos": state[16:][None, None],
                "arm_joint_pos": state[:7][None, None],
            },
            "language": {contract.language_key: [["pick and place"]]},
        }

        first = backend.infer_one(observation, seed=111)
        second = backend.infer_one(observation, seed=111)
        third = backend.infer_one(observation, seed=112)

        np.testing.assert_array_equal(first.z_rl, np.arange(4, dtype=np.float32))
        np.testing.assert_array_equal(first.z_rl, second.z_rl)
        np.testing.assert_array_equal(first.vla_reference_action, second.vla_reference_action)
        self.assertFalse(np.array_equal(first.vla_reference_action, third.vla_reference_action))
        np.testing.assert_array_equal(
            first.proprio, np.concatenate((state[7:16], state[16:]))
        )
        self.assertEqual(first.original_token_count, 3)
        self.assertEqual(first.selected_token_count, 3)
        self.assertEqual(policy.model.backbone_calls, 3)
        self.assertEqual(policy.model.action_head.encode_calls, 3)
        self.assertEqual(policy.model.action_head.action_calls, 3)
        self.assertEqual(policy.model.action_head.num_inference_timesteps, 7)
        self.assertTrue(torch.equal(encoder.packed, torch.arange(24).reshape(1, 3, 8)))

    def test_token_contract_rejects_random_and_signed_cache_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported token_sampling"):
            token_contract_fingerprint(
                token_scope="image",
                token_sampling="random",
                max_vl_tokens=192,
            )

        signed = types.SimpleNamespace(
            token_scope="image",
            token_sampling="uniform",
            max_vl_tokens=192,
        )
        with mock.patch(
            "groot_rlt.integration.lerobot_v3_offline_features.load_prefix_cache_contract",
            return_value=signed,
        ):
            with self.assertRaisesRegex(ValueError, "signed prefix-cache contract"):
                inspect_deployment_contract(
                    dataset_dir="/not/used",
                    model_path="/not/used",
                    processor_path="/not/used",
                    vlm_model_path="/not/used",
                    encoder_artifact="/not/used",
                    prefix_cache_manifest="/not/used",
                    info_payload={},
                    source_ego_key="ego",
                    source_wrist_key="wrist",
                    token_scope="all",
                    token_sampling="uniform",
                    max_vl_tokens=192,
                )


def _synthetic_info(*, height: int, width: int, frame_count: int) -> dict[str, Any]:
    scalar_features = {
        "observation.state": {"dtype": "float32", "shape": [26]},
        "action": {"dtype": "float32", "shape": [19]},
        "intervention": {"dtype": "bool", "shape": [1]},
        "teleop_stack.is_intervention": {"dtype": "bool", "shape": [1]},
        "teleop_stack.task_episode_id": {"dtype": "string", "shape": [1]},
        "teleop_stack.partition": {"dtype": "string", "shape": [1]},
        "teleop_stack.behavior_source": {"dtype": "string", "shape": [1]},
        "teleop_stack.advantage_label_rule": {"dtype": "string", "shape": [1]},
        "teleop_stack.terminal_label": {"dtype": "string", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    video_info = {
        "video.height": height,
        "video.width": width,
        "video.channels": 3,
        "video.fps": 10,
        "video.is_depth_map": False,
        "has_audio": False,
    }
    for key in ("observation.images.ego_view", "observation.images.wrist_view"):
        scalar_features[key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "info": dict(video_info),
        }
    return {
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "total_frames": frame_count,
        "total_tasks": 1,
        "fps": 10,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": scalar_features,
    }


class LocalV3FallbackTest(unittest.TestCase):
    @staticmethod
    def _write_video(av: Any, path: Path, frames: list[np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(path), mode="w")
        try:
            stream = container.add_stream("libx264", rate=10)
            stream.height, stream.width = frames[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            for array in frames:
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    def test_local_parquet_pyav_loader_validates_and_decodes(self) -> None:
        try:
            import av
            import pandas as pd
        except ImportError as exc:
            self.skipTest(str(exc))
        height, width, frame_count = 8, 10, 2
        info = _synthetic_info(height=height, width=width, frame_count=frame_count)
        camera_keys = ("observation.images.ego_view", "observation.images.wrist_view")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
            rows = []
            for frame_index in range(frame_count):
                done = frame_index == frame_count - 1
                rows.append(
                    {
                        "observation.state": _source_state(),
                        "action": _reference_action()[:19],
                        "intervention": False,
                        "teleop_stack.is_intervention": False,
                        "teleop_stack.task_episode_id": "task_episode_000",
                        "teleop_stack.partition": "policy_rollout",
                        "teleop_stack.behavior_source": "policy",
                        "teleop_stack.advantage_label_rule": "pending_value_function",
                        "teleop_stack.terminal_label": "success",
                        "next.reward": np.float32(1.0 if done else 0.0),
                        "next.done": done,
                        "timestamp": np.float32(frame_index / 10),
                        "frame_index": frame_index,
                        "episode_index": 0,
                        "index": frame_index,
                        "task_index": 0,
                    }
                )
            pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet")
            pd.DataFrame(
                [
                    {
                        "episode_index": 0,
                        "tasks": np.asarray(["pick and place"], dtype=object),
                        "length": frame_count,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                        "dataset_from_index": 0,
                        "dataset_to_index": frame_count,
                        **{f"videos/{key}/chunk_index": 0 for key in camera_keys},
                        **{f"videos/{key}/file_index": 0 for key in camera_keys},
                        **{f"videos/{key}/from_timestamp": 0.0 for key in camera_keys},
                        **{f"videos/{key}/to_timestamp": frame_count / 10 for key in camera_keys},
                    }
                ]
            ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
            pd.DataFrame([{"task_index": 0}]).to_parquet(root / "meta/tasks.parquet")
            frames = [
                np.full((height, width, 3), 40 + index * 100, dtype=np.uint8)
                for index in range(frame_count)
            ]
            try:
                for key in camera_keys:
                    self._write_video(
                        av,
                        root / f"videos/{key}/chunk-000/file-000.mp4",
                        frames,
                    )
            except Exception as exc:
                self.skipTest(f"test PyAV lacks an H.264 encoder: {exc}")

            loader = LocalV3ParquetPyAVLoader(root, info_payload=info, camera_keys=camera_keys)
            self.assertEqual(len(loader), frame_count)
            self.assertEqual(loader.num_episodes, 1)
            self.assertEqual(loader[0]["task"], "pick and place")
            for index in (1, 0):
                for key in camera_keys:
                    image = loader[index][key].numpy()
                    self.assertEqual(image.shape, (height, width, 3))
                    self.assertEqual(image.dtype, np.uint8)
            loader.close()

            missing = RuntimeError("LeRobot is not installed. test")
            missing.__cause__ = ImportError("test")
            with mock.patch(
                "groot_rlt.integration.lerobot_v3_offline_features.open_official_lerobot_dataset",
                side_effect=missing,
            ):
                auto_loader = open_v3_dataset_loader(
                    repo_id="local/test",
                    root=root,
                    info_payload=info,
                    camera_keys=camera_keys,
                    loader_mode="auto",
                )
            self.assertIsInstance(auto_loader, LocalV3ParquetPyAVLoader)
            auto_loader.close()

            (root / "data" / "unexpected.txt").write_text("bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected data parquet file path"):
                LocalV3ParquetPyAVLoader(root, info_payload=info, camera_keys=camera_keys)

    def test_auto_loader_prefers_injected_official_factory(self) -> None:
        sentinel = object()
        calls: list[dict[str, Any]] = []

        def factory(**kwargs: Any) -> object:
            calls.append(kwargs)
            return sentinel

        result = open_v3_dataset_loader(
            repo_id="local/test",
            root="/tmp/not-read",
            info_payload={},
            camera_keys=("ego",),
            dataset_factory=factory,
        )
        self.assertIs(result, sentinel)
        self.assertEqual(calls[0]["repo_id"], "local/test")
        self.assertFalse(calls[0]["download_videos"])


if __name__ == "__main__":
    unittest.main()
