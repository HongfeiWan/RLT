"""GR00T N1.7 backend for the RLT Machine-A feature contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from groot_rlt.integration.artifact_lineage import (
    canonical_json_sha256,
    checkpoint_fingerprint,
    file_sha256,
)
from groot_rlt.integration.nero_action_contract import (
    ACTOR_PROPRIO_CHANNEL_NAMES,
    ACTOR_PROPRIO_DIM,
    ROT6D_CONVENTION,
    VLA_REFERENCE_CHANNEL_NAMES,
    semantic_layout_hash,
)
from groot_rlt.integration.prefix_cache_contract import (
    PrefixCacheContract,
    load_prefix_cache_contract,
    require_sha256,
    validate_prefix_cache_deployment_paths,
    vlm_content_fingerprint,
)
from groot_rlt.representation.encoder_artifact import (
    ENCODER_ARCHITECTURE,
    ENCODER_ARTIFACT_KIND,
    FEATURE_TAP,
    PROCESSOR_MODE,
    SOURCE_ARCHITECTURE,
    load_encoder_ema_artifact,
    tensor_state_sha256,
)

_LINEAGE_KEYS = {
    "cache_fingerprint",
    "checkpoint_fingerprint",
    "feature_tap",
    "processor_mode",
}


@dataclass(frozen=True)
class LoadedServingEncoder:
    """Encoder plus the immutable metadata advertised in the Machine-A handshake."""

    encoder: Any
    checkpoint_args: dict[str, Any]
    handshake: dict[str, Any]


def _validate_representation_lineage(
    checkpoint_args: Mapping[str, Any],
    *,
    expected_checkpoint_fingerprint: str,
    expected_cache_fingerprint: str,
) -> dict[str, Any]:
    lineage = checkpoint_args.get("representation_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(
            "Legacy full checkpoint lacks args.representation_lineage; "
            "historical schema1 checkpoints cannot be served against a schema2 prefix"
        )
    lineage = dict(lineage)
    missing = sorted(_LINEAGE_KEYS - set(lineage))
    unexpected = sorted(set(lineage) - _LINEAGE_KEYS)
    if missing or unexpected:
        raise ValueError(
            "Legacy checkpoint representation_lineage keys mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if lineage["checkpoint_fingerprint"] != expected_checkpoint_fingerprint:
        raise ValueError("Legacy checkpoint 400k fingerprint does not match deployment")
    if lineage["cache_fingerprint"] != expected_cache_fingerprint:
        raise ValueError("Legacy checkpoint prefix-cache fingerprint does not match deployment")
    if lineage["feature_tap"] != FEATURE_TAP or lineage["processor_mode"] != PROCESSOR_MODE:
        raise ValueError("Legacy checkpoint was trained from a non-serving-equivalent prefix")
    return lineage


def _encoder_state_from_full_model(encoder: Any) -> dict[str, Any]:
    state = encoder.state_dict()
    selected = {
        name: value
        for name, value in state.items()
        if name in {"query_token", "encoder_memory_pos"}
        or name.startswith("input_proj.")
        or name.startswith("encoder.")
    }
    if not selected:
        raise ValueError("Legacy checkpoint has no strict encoder state")
    return selected


def load_serving_rl_token_encoder(
    *,
    encoder_artifact: str | Path | None,
    legacy_full_checkpoint: str | Path | None,
    expected_checkpoint_fingerprint: str,
    expected_cache_fingerprint: str,
    cache_contract: PrefixCacheContract,
    device: Any,
) -> LoadedServingEncoder:
    """Load exactly one explicit encoder source; never fall back after a load error."""

    expected_checkpoint_fingerprint = require_sha256(
        expected_checkpoint_fingerprint, "expected_checkpoint_fingerprint"
    )
    expected_cache_fingerprint = require_sha256(
        expected_cache_fingerprint, "expected_cache_fingerprint"
    )
    if (encoder_artifact is None) == (legacy_full_checkpoint is None):
        raise ValueError(
            "Exactly one of encoder_artifact or legacy_full_checkpoint must be provided"
        )
    if cache_contract.checkpoint_fingerprint != expected_checkpoint_fingerprint:
        raise ValueError("Cache contract and expected 400k fingerprints differ")
    if cache_contract.fingerprint != expected_cache_fingerprint:
        raise ValueError("Cache contract and expected cache fingerprints differ")

    if encoder_artifact is not None:
        artifact_path = Path(encoder_artifact).expanduser().resolve()
        loaded = load_encoder_ema_artifact(
            artifact_path,
            expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
            expected_cache_fingerprint=expected_cache_fingerprint,
            device=device,
        )
        encoder = loaded.encoder
        if encoder.config.input_dim != cache_contract.input_dim:
            raise ValueError(
                f"Encoder input_dim={encoder.config.input_dim} differs from prefix-cache "
                f"input_dim={cache_contract.input_dim}"
            )
        if encoder.config.max_vl_tokens != cache_contract.max_vl_tokens:
            raise ValueError(
                f"Encoder max_vl_tokens={encoder.config.max_vl_tokens} differs from "
                f"prefix-cache max_vl_tokens={cache_contract.max_vl_tokens}"
            )
        manifest = json.loads(loaded.manifest_path.read_text(encoding="utf-8"))
        handshake = {
            "source_kind": "encoder_ema_artifact",
            "artifact_kind": ENCODER_ARTIFACT_KIND,
            "architecture": ENCODER_ARCHITECTURE,
            "source_architecture": SOURCE_ARCHITECTURE,
            "artifact_path": str(loaded.artifact_path),
            "artifact_manifest_path": str(loaded.manifest_path),
            "artifact_fingerprint": manifest["metadata_sha256"],
            "artifact_file_sha256": manifest["artifact_sha256"],
            "encoder_tensor_sha256": manifest["encoder_state_sha256"],
            "representation_checkpoint_file_sha256": manifest["source_checkpoint_sha256"],
        }
        checkpoint_args: dict[str, Any] = {}
    else:
        # Importing this path is intentionally explicit: it allocates the decoder and
        # unpickles the full trusted training checkpoint. Artifact errors never reach it.
        import torch

        from groot_rlt.representation.visualize_rl_token_umap import load_rl_token_encoder

        checkpoint_path = Path(legacy_full_checkpoint).expanduser().resolve()
        encoder, checkpoint_args, _, _ = load_rl_token_encoder(
            checkpoint_path,
            torch.device(device),
        )
        _validate_representation_lineage(
            checkpoint_args,
            expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
            expected_cache_fingerprint=expected_cache_fingerprint,
        )
        for name, expected in (
            ("token_scope", cache_contract.token_scope),
            ("token_sampling", cache_contract.token_sampling),
            ("max_vl_tokens", cache_contract.max_vl_tokens),
        ):
            if checkpoint_args.get(name) != expected:
                raise ValueError(
                    f"Legacy checkpoint {name}={checkpoint_args.get(name)!r} differs "
                    f"from prefix cache {expected!r}"
                )
        if encoder.config.input_dim != cache_contract.input_dim:
            raise ValueError("Legacy encoder input dimension differs from prefix cache")
        checkpoint_file_sha256 = f"sha256:{file_sha256(checkpoint_path)}"
        encoder_tensor_sha256 = tensor_state_sha256(_encoder_state_from_full_model(encoder))
        legacy_material = {
            "architecture": SOURCE_ARCHITECTURE,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "encoder_tensor_sha256": encoder_tensor_sha256,
            "checkpoint_fingerprint": expected_checkpoint_fingerprint,
            "cache_fingerprint": expected_cache_fingerprint,
            "representation_checkpoint_schema_version": 2,
            "prefix_cache_schema_version": 2,
        }
        handshake = {
            "source_kind": "legacy_full_checkpoint",
            "artifact_kind": "groot_rlt.legacy_full_training_checkpoint",
            "architecture": SOURCE_ARCHITECTURE,
            "source_architecture": SOURCE_ARCHITECTURE,
            "artifact_path": str(checkpoint_path),
            "artifact_manifest_path": None,
            "artifact_fingerprint": canonical_json_sha256(legacy_material),
            "artifact_file_sha256": checkpoint_file_sha256,
            "encoder_tensor_sha256": encoder_tensor_sha256,
            "representation_checkpoint_file_sha256": checkpoint_file_sha256,
        }

    handshake.update(
        {
            "checkpoint_fingerprint": expected_checkpoint_fingerprint,
            "cache_fingerprint": expected_cache_fingerprint,
            "representation_checkpoint_schema_version": 2,
            "prefix_cache_schema_version": 2,
            "feature_tap": cache_contract.feature_tap,
            "processor_mode": cache_contract.processor_mode,
            "prefix_cache_manifest_path": cache_contract.manifest_path,
            "token_scope": cache_contract.token_scope,
            "token_sampling": cache_contract.token_sampling,
            "max_vl_tokens": cache_contract.max_vl_tokens,
            "input_dim": cache_contract.input_dim,
            "z_dim": int(encoder.config.rl_token_dim),
            "video_modality_keys": list(cache_contract.video_modality_keys),
        }
    )
    return LoadedServingEncoder(
        encoder=encoder,
        checkpoint_args=checkpoint_args,
        handshake=handshake,
    )


class FeatureBackend(Protocol):
    """Backend that produces one Machine-A payload from one observation."""

    def infer_one(self, observation: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FeatureContract:
    z_dim: int = 2048
    chunk_len: int = 10
    action_dim: int = 26
    proprio_dim: int = 19

    def validate(self) -> None:
        for name in ("z_dim", "chunk_len", "action_dim", "proprio_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


def channel_layout_hash(
    names: list[str] | tuple[str, ...],
    *,
    rotation_convention: str | None = None,
) -> str:
    """Return a stable fingerprint for an ordered per-dimension layout."""

    if rotation_convention is not None:
        return semantic_layout_hash(names, rotation_convention=rotation_convention)
    payload = json.dumps(list(names), ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expand_layout(keys: tuple[str, ...], arrays: list[np.ndarray]) -> list[str]:
    return [
        f"{key}[{index}]" for key, array in zip(keys, arrays) for index in range(array.shape[-1])
    ]


class MachineAFeaturePolicy:
    """Validate GR00T output against the existing online-RL wire contract."""

    def __init__(
        self,
        backend: FeatureBackend,
        contract: FeatureContract,
        *,
        supports_batch: bool = True,
    ):
        contract.validate()
        self.backend = backend
        self.contract = contract
        self.supports_batch = supports_batch

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        z_rl = np.asarray(payload["z_rl"], dtype=np.float32)
        if z_rl.ndim == 2 and z_rl.shape[0] == 1:
            z_rl = z_rl[0]
        if z_rl.shape != (self.contract.z_dim,):
            raise ValueError(f"z_rl must have shape ({self.contract.z_dim},), got {z_rl.shape}")

        ref_chunk = np.asarray(payload["ref_chunk"], dtype=np.float32)
        if ref_chunk.ndim == 3 and ref_chunk.shape[0] == 1:
            ref_chunk = ref_chunk[0]
        if (
            ref_chunk.ndim != 2
            or ref_chunk.shape[0] < self.contract.chunk_len
            or ref_chunk.shape[1] != self.contract.action_dim
        ):
            raise ValueError(
                "ref_chunk must have exact action dimension and sufficient horizon: "
                f"expected [>={self.contract.chunk_len}, {self.contract.action_dim}], "
                f"got {ref_chunk.shape}"
            )
        ref_chunk = ref_chunk[: self.contract.chunk_len]

        proprio = np.asarray(payload["proprio"], dtype=np.float32)
        if proprio.ndim == 2 and proprio.shape[0] == 1:
            proprio = proprio[0]
        if proprio.shape != (self.contract.proprio_dim,):
            raise ValueError(
                f"proprio must have shape ({self.contract.proprio_dim},), got {proprio.shape}"
            )

        for name, expected_dim in (
            ("action_layout", self.contract.action_dim),
            ("proprio_layout", self.contract.proprio_dim),
        ):
            if name in payload and len(payload[name]) != expected_dim:
                raise ValueError(f"{name} must contain {expected_dim} names")

        return {**payload, "z_rl": z_rl, "ref_chunk": ref_chunk, "proprio": proprio}

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        if "batch" in request:
            if not self.supports_batch:
                raise ValueError("this GR00T feature backend does not support batched inference")
            observations = request["batch"]
            if not isinstance(observations, list):
                raise TypeError("batch request must contain a list of observations")
            return {
                "batch_results": [
                    self._normalize(self.backend.infer_one(observation))
                    for observation in observations
                ]
            }
        return self._normalize(self.backend.infer_one(request))


class GrootN1d7FeatureBackend:
    """Run GR00T action inference and RL-token encoding in one backbone pass.

    The returned ``ref_chunk`` is in processor-decoded physical action space.
    Normalization remains the responsibility of ``ActionRepresentationAdapter``
    in the online learner, matching the existing openpi Machine-A boundary.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        processor_path: str | Path,
        vlm_model_path: str | Path,
        prefix_cache_manifest: str | Path,
        expected_checkpoint_fingerprint: str,
        expected_cache_fingerprint: str,
        expected_vlm_content_fingerprint: str,
        embodiment_tag: str,
        device: str,
        contract: FeatureContract,
        rl_token_encoder_artifact: str | Path | None = None,
        legacy_rl_token_checkpoint: str | Path | None = None,
        strict: bool = True,
        token_scope: str | None = None,
        token_sampling: str | None = None,
        max_vl_tokens: int | None = None,
        proprio_keys: tuple[str, ...] | None = None,
        image_key_map: dict[str, str] | None = None,
        flat_state_layout: tuple[tuple[str, int], ...] | None = None,
        default_instruction: str | None = None,
        num_inference_timesteps: int | None = None,
    ) -> None:
        import torch

        from groot_rlt.representation.precompute_rl_tokens_and_vla_actions import (
            PrecomputeCheckpointRokaePolicy,
        )
        from groot_rlt.representation.train_vl_embedding_autoencoder import pack_vl_tokens

        contract.validate()
        self.contract = contract
        self.device = torch.device(device)
        expected_checkpoint_fingerprint = require_sha256(
            expected_checkpoint_fingerprint,
            "expected_checkpoint_fingerprint",
        )
        expected_cache_fingerprint = require_sha256(
            expected_cache_fingerprint,
            "expected_cache_fingerprint",
        )
        expected_vlm_content_fingerprint = require_sha256(
            expected_vlm_content_fingerprint,
            "expected_vlm_content_fingerprint",
        )
        cache_contract = load_prefix_cache_contract(
            prefix_cache_manifest,
            expected_cache_fingerprint=expected_cache_fingerprint,
            expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
        )
        model_path, processor_path, vlm_model_path = validate_prefix_cache_deployment_paths(
            cache_contract,
            model_path=model_path,
            processor_path=processor_path,
            vlm_model_path=vlm_model_path,
            context="Serving",
        )
        actual_checkpoint_fingerprint, _ = checkpoint_fingerprint(model_path)
        if actual_checkpoint_fingerprint != expected_checkpoint_fingerprint:
            raise ValueError(
                "Serving GR00T checkpoint fingerprint mismatch: "
                f"model={actual_checkpoint_fingerprint} "
                f"expected={expected_checkpoint_fingerprint}"
            )
        actual_vlm_content_fingerprint = vlm_content_fingerprint(vlm_model_path)
        if actual_vlm_content_fingerprint != expected_vlm_content_fingerprint:
            raise ValueError(
                "Serving VLM deployment content fingerprint mismatch: "
                f"actual={actual_vlm_content_fingerprint} "
                f"expected={expected_vlm_content_fingerprint}"
            )
        self.policy = PrecomputeCheckpointRokaePolicy(
            model_path=model_path,
            processor_path=processor_path,
            device=str(self.device),
            strict=strict,
            vlm_model_path=vlm_model_path,
            embodiment_tag=embodiment_tag,
        )
        loaded_encoder = load_serving_rl_token_encoder(
            encoder_artifact=rl_token_encoder_artifact,
            legacy_full_checkpoint=legacy_rl_token_checkpoint,
            expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
            expected_cache_fingerprint=expected_cache_fingerprint,
            cache_contract=cache_contract,
            device=self.device,
        )
        encoder = loaded_encoder.encoder
        checkpoint_args = loaded_encoder.checkpoint_args
        if encoder.config.rl_token_dim != contract.z_dim:
            raise ValueError(
                f"RL-token encoder dim {encoder.config.rl_token_dim} != z_dim {contract.z_dim}"
            )
        self.encoder = encoder
        if token_scope is not None and token_scope != cache_contract.token_scope:
            raise ValueError(
                f"serving token_scope={token_scope!r} differs from encoder training "
                f"token_scope={cache_contract.token_scope!r}"
            )
        if token_sampling is not None and token_sampling != cache_contract.token_sampling:
            raise ValueError(
                f"serving token_sampling={token_sampling!r} differs from encoder training "
                f"token_sampling={cache_contract.token_sampling!r}"
            )
        if max_vl_tokens is not None and int(max_vl_tokens) != cache_contract.max_vl_tokens:
            raise ValueError(
                f"serving max_vl_tokens={max_vl_tokens} differs from encoder training "
                f"max_vl_tokens={cache_contract.max_vl_tokens}"
            )
        self.token_scope = cache_contract.token_scope
        self.token_sampling = cache_contract.token_sampling
        self.max_vl_tokens = cache_contract.max_vl_tokens
        backbone_dim = getattr(self.policy.model.config, "backbone_embedding_dim", None)
        if backbone_dim is None:
            raise ValueError("GR00T model config does not expose backbone_embedding_dim")
        if int(backbone_dim) != int(cache_contract.input_dim):
            raise ValueError(
                f"Prefix-cache input_dim={cache_contract.input_dim} does not match "
                f"GR00T backbone_embedding_dim={backbone_dim}"
            )
        policy_video_keys = tuple(self.policy.modality_configs["video"].modality_keys)
        if policy_video_keys != cache_contract.video_modality_keys:
            raise ValueError(
                f"Serving GR00T video keys={policy_video_keys!r} differ from prefix-cache "
                f"video keys={cache_contract.video_modality_keys!r}"
            )
        self.rl_token_checkpoint_args = checkpoint_args
        self.rl_token_serving_metadata = {
            **loaded_encoder.handshake,
            "model_path": str(model_path),
            "processor_path": str(processor_path),
            "vlm_model_path": str(vlm_model_path),
            "vlm_deployment_content_fingerprint": actual_vlm_content_fingerprint,
            "vlm_fingerprint_scope": "deployment_only_not_representation_training_lineage",
        }
        checkpoint_state_keys = tuple(self.policy.modality_configs["state"].modality_keys)
        if (
            proprio_keys is None
            and checkpoint_state_keys == ("eef_9d", "hand_joint_pos", "arm_joint_pos")
            and contract.proprio_dim == ACTOR_PROPRIO_DIM
        ):
            # The complete state still reaches the frozen 400k model. Only this
            # projected view is advertised to actor/critic consumers.
            self.proprio_keys = ("eef_9d", "hand_joint_pos")
        else:
            self.proprio_keys = proprio_keys
        self.image_key_map = dict(image_key_map or {})
        self.flat_state_layout = flat_state_layout
        self.default_instruction = default_instruction
        action_keys = tuple(self.policy.modality_configs["action"].modality_keys)
        self.action_layout: list[str] | None = None
        self.rot6d_convention: str | None = None
        if action_keys == ("eef_9d", "hand_joint_target", "arm_joint_target"):
            self.action_layout = list(VLA_REFERENCE_CHANNEL_NAMES)
            self.rot6d_convention = ROT6D_CONVENTION
        self.proprio_layout: list[str] | None = None
        try:
            state_layout = self._resolved_flat_state_layout()
        except ValueError:
            state_layout = None
        if state_layout is not None:
            selected_keys = self.proprio_keys or tuple(key for key, _ in state_layout)
            dimensions = dict(state_layout)
            if all(key in dimensions for key in selected_keys):
                self.proprio_layout = [
                    f"{key}[{index}]" for key in selected_keys for index in range(dimensions[key])
                ]
        if (
            checkpoint_state_keys == ("eef_9d", "hand_joint_pos", "arm_joint_pos")
            and contract.proprio_dim == ACTOR_PROPRIO_DIM
            and self.proprio_layout != list(ACTOR_PROPRIO_CHANNEL_NAMES)
        ):
            raise ValueError("Nero 19D proprio layout must be exactly eef9+hand10")
        self.action_layout_hash = (
            None
            if self.action_layout is None
            else channel_layout_hash(
                self.action_layout,
                rotation_convention=self.rot6d_convention,
            )
        )
        self.proprio_layout_hash = (
            None
            if self.proprio_layout is None
            else channel_layout_hash(
                self.proprio_layout,
                rotation_convention=self.rot6d_convention,
            )
        )
        self._pack_vl_tokens = pack_vl_tokens
        self._torch = torch
        if num_inference_timesteps is not None:
            if int(num_inference_timesteps) <= 0:
                raise ValueError("num_inference_timesteps must be positive")
            self.policy.model.action_head.num_inference_timesteps = int(num_inference_timesteps)
        self.num_inference_timesteps = int(self.policy.model.action_head.num_inference_timesteps)
        self._validate_capabilities()

    def _validate_capabilities(self) -> None:
        model = self.policy.model
        missing = [
            name
            for name in ("prepare_input", "backbone", "action_head")
            if not hasattr(model, name)
        ]
        action_head = getattr(model, "action_head", None)
        for name in ("_encode_features", "get_action_with_features"):
            if action_head is None or not hasattr(action_head, name):
                missing.append(f"action_head.{name}")
        if missing:
            raise RuntimeError(
                "The selected GR00T checkout/checkpoint lacks required N1.7 feature APIs: "
                + ", ".join(missing)
            )

    @staticmethod
    def _batched_video(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.uint8)
        if array.ndim == 3:
            array = array[None, None]
        elif array.ndim == 4:
            array = array[None]
        if array.ndim != 5 or array.shape[0] != 1:
            raise ValueError(
                "Machine-A single observations require video HWC, THWC, or [1,T,H,W,C]; "
                f"got {array.shape}"
            )
        return array

    @staticmethod
    def _batched_state(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, None]
        elif array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(
                f"Machine-A single observations require state D, TD, or [1,T,D]; got {array.shape}"
            )
        return array

    def _resolved_flat_state_layout(self) -> tuple[tuple[str, int], ...]:
        if self.flat_state_layout is not None:
            return self.flat_state_layout
        keys = tuple(self.policy.modality_configs["state"].modality_keys)
        if keys == ("eef_9d", "hand_joint_pos", "arm_joint_pos"):
            return (("eef_9d", 9), ("hand_joint_pos", 10), ("arm_joint_pos", 7))
        if len(keys) == 1:
            return ((keys[0], self.contract.proprio_dim),)
        raise ValueError(
            "flat observation state requires --flat-state-field KEY=DIM for every GR00T "
            f"state key; checkpoint expects {keys}"
        )

    def _adapt_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Accept native GR00T observations or the legacy flat RLT wire shape."""

        expected_video = tuple(self.policy.modality_configs["video"].modality_keys)
        expected_state = tuple(self.policy.modality_configs["state"].modality_keys)
        language_key = self.policy.language_key

        if all(isinstance(observation.get(key), dict) for key in ("video", "state", "language")):
            videos = {key: self._batched_video(observation["video"][key]) for key in expected_video}
            states = {key: self._batched_state(observation["state"][key]) for key in expected_state}
            language = observation["language"][language_key]
            if isinstance(language, str):
                language = [[language]]
            elif language and isinstance(language[0], str):
                language = [list(language)]
            return {"video": videos, "state": states, "language": {language_key: language}}

        if "images" not in observation or "state" not in observation:
            raise ValueError(
                "observation must use native GR00T {video,state,language} or flat RLT "
                "{images,state,prompt} format"
            )
        source_images = observation["images"]
        if not isinstance(source_images, dict):
            raise TypeError("flat observation images must be a mapping")
        inverse_image_map = {target: source for source, target in self.image_key_map.items()}
        videos = {}
        for target in expected_video:
            source = inverse_image_map.get(target, target)
            if source not in source_images:
                raise KeyError(
                    f"missing image {source!r} for GR00T video key {target!r}; "
                    "use --image-key SOURCE=TARGET"
                )
            videos[target] = self._batched_video(source_images[source])

        flat_state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        layout = self._resolved_flat_state_layout()
        if tuple(key for key, _ in layout) != expected_state:
            raise ValueError(
                f"flat state layout keys {tuple(key for key, _ in layout)} do not match "
                f"checkpoint keys {expected_state}"
            )
        if sum(dim for _, dim in layout) != flat_state.size:
            raise ValueError(
                f"flat state layout totals {sum(dim for _, dim in layout)} values, "
                f"observation has {flat_state.size}"
            )
        states = {}
        offset = 0
        for key, dim in layout:
            states[key] = flat_state[offset : offset + dim][None, None]
            offset += dim
        instruction = observation.get("prompt", self.default_instruction)
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(
                "flat observation requires a non-empty prompt or --default-instruction"
            )
        return {
            "video": videos,
            "state": states,
            "language": {language_key: [[instruction]]},
        }

    def _prepare_policy_inputs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
        from gr00t.data.types import MessageType, VLAStepData

        observation = self._adapt_observation(observation)
        if self.policy.strict:
            self.policy.check_observation(observation)
        unbatched = self.policy._unbatch_observation(observation)
        processed = []
        states = []
        for item in unbatched:
            states.append(item["state"])
            step = VLAStepData(
                images=item["video"],
                states=item["state"],
                actions={},
                text=item["language"][self.policy.language_key][0],
                embodiment=self.policy.embodiment_tag,
            )
            processed.append(
                self.policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            )
        collated = self.policy.collate_fn(processed)
        collated = self.policy._rec_to_dtype(collated, dtype=self._torch.bfloat16)
        return collated, states

    def _extract_proprio(self, states: dict[str, np.ndarray]) -> np.ndarray:
        keys = self.proprio_keys or tuple(self.policy.modality_configs["state"].modality_keys)
        missing = [key for key in keys if key not in states]
        if missing:
            raise KeyError(f"GR00T observation is missing configured proprio keys: {missing}")
        parts = []
        for key in keys:
            value = np.asarray(states[key], dtype=np.float32)
            if value.ndim == 1:
                parts.append(value)
            elif value.ndim == 2:
                parts.append(value[-1])
            else:
                raise ValueError(f"state[{key!r}] must be rank-1 or rank-2, got {value.shape}")
        proprio = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
        if proprio.size != self.contract.proprio_dim:
            raise ValueError(
                f"configured state keys provide {proprio.size} values, "
                f"but proprio_dim={self.contract.proprio_dim}"
            )
        return proprio

    def infer_one(self, observation: dict[str, Any]) -> dict[str, Any]:
        collated, states = self._prepare_policy_inputs(observation)
        if len(states) != 1:
            raise ValueError(
                "Machine-A single inference expects batch size 1; use the outer batch request "
                "for multiple observations"
            )

        model = self.policy.model
        with self._torch.inference_mode():
            backbone_inputs, action_inputs = model.prepare_input(**collated)
            backbone_output = model.backbone(backbone_inputs)
            # The RL-token encoder was trained on the raw backbone embeddings.
            # ``_encode_features`` applies VLLN/self-attention and mutates
            # ``backbone_output`` in place, so capture those raw tokens first.
            packed, packed_mask, _, _, _ = self._pack_vl_tokens(
                backbone_output,
                token_scope=self.token_scope,
                max_tokens=self.max_vl_tokens,
                token_sampling=self.token_sampling,
            )
            features = model.action_head._encode_features(backbone_output, action_inputs)
            model_pred = model.action_head.get_action_with_features(
                backbone_features=features.backbone_features,
                state_features=features.state_features,
                embodiment_id=action_inputs.embodiment_id,
                backbone_output=backbone_output,
                action_input=action_inputs,
                options=None,
            )
            z_rl = self.encoder.encode_rl_token(packed.float(), packed_mask)

        normalized_action = model_pred["action_pred"].float().cpu().numpy()
        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.policy.modality_configs["state"].modality_keys
        }
        decoded = self.policy.processor.decode_action(
            normalized_action,
            self.policy.embodiment_tag,
            batched_states,
        )
        action_keys = tuple(self.policy.modality_configs["action"].modality_keys)
        decoded_parts = [np.asarray(decoded[key][0], dtype=np.float32) for key in action_keys]
        ref_chunk = np.concatenate(decoded_parts, axis=-1)
        if ref_chunk.shape[0] < self.contract.chunk_len:
            raise ValueError(
                f"GR00T action horizon {ref_chunk.shape[0]} < chunk_len {self.contract.chunk_len}"
            )
        if ref_chunk.shape[1] != self.contract.action_dim:
            raise ValueError(
                f"GR00T action dim {ref_chunk.shape[1]} != action_dim {self.contract.action_dim}"
            )
        proprio_parts = [
            np.asarray(states[0][key], dtype=np.float32)
            for key in (self.proprio_keys or self.policy.modality_configs["state"].modality_keys)
        ]
        action_layout = getattr(self, "action_layout", None) or _expand_layout(
            action_keys, decoded_parts
        )
        proprio_keys = tuple(
            self.proprio_keys or self.policy.modality_configs["state"].modality_keys
        )
        if len(action_layout) != ref_chunk.shape[1]:
            raise RuntimeError("configured action layout does not match decoded action dimension")
        proprio_layout = getattr(self, "proprio_layout", None) or _expand_layout(
            proprio_keys, proprio_parts
        )
        if len(proprio_layout) != self.contract.proprio_dim:
            raise RuntimeError("configured proprio layout does not match proprio dimension")
        return {
            "z_rl": z_rl[0].detach().float().cpu().numpy(),
            "ref_chunk": ref_chunk,
            "proprio": self._extract_proprio(states[0]),
            "action_keys": action_keys,
            "action_layout": action_layout,
            "action_layout_hash": channel_layout_hash(
                action_layout,
                rotation_convention=getattr(self, "rot6d_convention", None),
            ),
            "proprio_layout": proprio_layout,
            "proprio_layout_hash": channel_layout_hash(
                proprio_layout,
                rotation_convention=getattr(self, "rot6d_convention", None),
            ),
            "rot6d_convention": getattr(self, "rot6d_convention", None),
            "action_space": "processor_decoded_physical",
            "num_inference_timesteps": self.num_inference_timesteps,
            "rtc_applied": False,
        }
