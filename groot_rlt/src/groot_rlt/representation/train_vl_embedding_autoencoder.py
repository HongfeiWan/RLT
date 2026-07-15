#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a compact RL-token autoencoder over GR00T N1.7 VL embeddings.

This script builds only the representation-learning piece needed before online RLT:

1. Run the frozen GR00T/Cosmos VLA backbone and collect ``backbone_features``.
2. Encode the valid VL token sequence with a learned query token.
3. Expose the query output as ``encode_rl_token(...)`` for a future actor/critic.
4. Decode the token with teacher forcing and reconstruct the frozen VL embeddings.

The online actor/learner rollout loop, stride replay-buffer expansion, critical-phase
router, and reference-action dropout are intentionally left outside this script.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from groot_rlt.groot_repo import ensure_groot_repo
from groot_rlt.integration.artifact_lineage import (
    canonical_json_sha256,
    checkpoint_fingerprint,
    file_sha256,
)
from groot_rlt.paths import VL_EMBEDDING_CACHE_DIR

REPO_ROOT = ensure_groot_repo()
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: E402

from groot_rlt.integration.checkpoint_policy_utils import resolve_processor_path  # noqa: E402

try:  # noqa: E402
    from groot_rlt.integration.defaults import (
        L10_BASE_MODEL_PATH,
        L10_MODALITY_CONFIG_PATH,
        L10_PREPARED_DATASET_DIR,
        L10_VLM_MODEL_PATH,
    )
except Exception:  # pragma: no cover - keeps the script usable outside IsaacLab examples.
    L10_BASE_MODEL_PATH = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B"
    L10_MODALITY_CONFIG_PATH = (
        REPO_ROOT / "examples" / "IsaacLab" / "rokae_xmate3_l10_multiview_modality_config.py"
    )
    L10_PREPARED_DATASET_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "trimmed"
    L10_VLM_MODEL_PATH = REPO_ROOT / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"


TokenScope = Literal["all", "image", "non_image"]
TokenSampling = Literal["head", "tail", "uniform", "random"]

RL_TOKEN_ARCHITECTURE = "openpi_rlt_strict_cross_attention_v1"
RL_TOKEN_CHECKPOINT_SCHEMA_VERSION = 2
VL_CACHE_SCHEMA_VERSION = 2
VL_CACHE_FEATURE_TAP = "raw_backbone_pre_action_head"
VL_CACHE_PROCESSOR_MODE = "eval"


@dataclass
class VLTokenAutoencoderConfig:
    input_dim: int = 2048
    model_dim: int = 2048
    rl_token_dim: int = 2048
    max_vl_tokens: int = 512
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_prefix_mask_token: bool = False
    use_decoder_cross_attention: bool = True


def sinusoidal_position_embeddings(seq_len: int, dim: int) -> torch.Tensor:
    """Match openpi's trainable sinusoidal parameter initialization."""
    if dim % 2 != 0:
        raise ValueError(f"Position embedding dimension must be even, got {dim}")
    position = torch.arange(seq_len, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    return torch.cat(
        [
            torch.sin(position[:, None] * div_term),
            torch.cos(position[:, None] * div_term),
        ],
        dim=-1,
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        attn_mask = None
        seq_len = x.shape[1]
        if key_padding_mask is not None or causal:
            attn_mask = torch.zeros(
                x.shape[0],
                1,
                seq_len,
                seq_len,
                dtype=query.dtype,
                device=x.device,
            )
            if key_padding_mask is not None:
                attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                attn_mask = attn_mask.masked_fill(causal_mask[None, None, :, :], -torch.inf)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(x.shape[0], x.shape[1], -1)
        return self.o_proj(output)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(memory))
        value = self._split_heads(self.v_proj(memory))

        attn_mask = None
        if memory_key_padding_mask is not None:
            attn_mask = torch.zeros(
                x.shape[0],
                1,
                x.shape[1],
                memory.shape[1],
                dtype=query.dtype,
                device=x.device,
            )
            attn_mask = attn_mask.masked_fill(memory_key_padding_mask[:, None, None, :], -torch.inf)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(x.shape[0], x.shape[1], -1)
        return self.o_proj(output)


class OpenPiGeGLUMLP(nn.Module):
    """PyTorch equivalent of openpi's Dense, Dropout, GeGLU, Dense MLP."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.geglu_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.input_proj(x))
        value, gate = self.geglu_proj(x).chunk(2, dim=-1)
        return self.output_proj(value * F.gelu(gate, approximate="tanh"))


class OpenPiCrossAttentionBlock(nn.Module):
    """Exact block order used by openpi-RLT's CrossAttentionLayer."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.mlp = OpenPiGeGLUMLP(dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_norm(x))
        x = x + self.cross_attn(
            self.cross_attn_norm(x),
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return x + self.mlp(self.mlp_norm(x))


class VLTokenAutoencoder(nn.Module):
    """Strict PyTorch reproduction of openpi-RLT's RLTokenModel."""

    def __init__(self, config: VLTokenAutoencoderConfig):
        super().__init__()
        self.config = config
        if config.rl_token_dim != config.model_dim:
            raise ValueError(
                "The openpi-style RL token is the query output and must match model_dim; "
                f"got rl_token_dim={config.rl_token_dim}, model_dim={config.model_dim}."
            )
        if not config.use_decoder_cross_attention:
            raise ValueError("Strict openpi-RLT reproduction requires decoder cross-attention.")

        self.input_proj = (
            nn.Linear(config.input_dim, config.model_dim)
            if config.input_dim != config.model_dim
            else nn.Identity()
        )
        self.query_token = nn.Parameter(sinusoidal_position_embeddings(1, config.model_dim))
        self.encoder_memory_pos = nn.Parameter(
            sinusoidal_position_embeddings(config.max_vl_tokens, config.model_dim)
        )
        self.decoder_query = nn.Parameter(
            sinusoidal_position_embeddings(config.max_vl_tokens, config.model_dim)
        )
        self.decoder_memory_pos = nn.Parameter(sinusoidal_position_embeddings(1, config.model_dim))
        self.encoder = nn.ModuleList(
            [
                OpenPiCrossAttentionBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                OpenPiCrossAttentionBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self.output_proj = (
            nn.Linear(config.model_dim, config.input_dim)
            if config.input_dim != config.model_dim
            else nn.Identity()
        )
        if config.use_prefix_mask_token:
            self.prefix_mask_token = nn.Parameter(torch.empty(1, 1, config.model_dim))
            nn.init.normal_(self.prefix_mask_token, std=0.02)
        else:
            self.register_parameter("prefix_mask_token", None)

    def encode_rl_token(self, vl_embeddings: torch.Tensor, vl_mask: torch.Tensor) -> torch.Tensor:
        """Return the compact token intended for future RL actor/critic conditioning.

        Args:
            vl_embeddings: Tensor of shape ``[B, S, input_dim]`` from the frozen VLA.
            vl_mask: Boolean tensor of shape ``[B, S]`` where True marks valid tokens.

        Returns:
            Tensor of shape ``[B, 2048]``.
        """
        if vl_embeddings.ndim != 3:
            raise ValueError(
                f"Expected vl_embeddings shape [B, S, D], got {tuple(vl_embeddings.shape)}"
            )
        batch_size, seq_len, _ = vl_embeddings.shape
        if tuple(vl_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                f"Expected vl_mask shape {(batch_size, seq_len)}, got {tuple(vl_mask.shape)}"
            )
        empty_samples = torch.nonzero(~vl_mask.bool().any(dim=1), as_tuple=False).flatten()
        if empty_samples.numel():
            raise ValueError(
                "Every RL-token sample must contain at least one valid prefix token; "
                f"empty batch indices={empty_samples.tolist()}"
            )
        if seq_len > self.config.max_vl_tokens:
            raise ValueError(
                f"Expected at most {self.config.max_vl_tokens} tokens, got {seq_len}. "
                "Pack or subsample the VL sequence before calling encode_rl_token()."
            )

        x = self.query_token.expand(batch_size, -1, -1)
        memory = self.input_proj(vl_embeddings)
        memory = memory + self.encoder_memory_pos[:seq_len].unsqueeze(0)
        for block in self.encoder:
            x = block(x, memory, memory_key_padding_mask=~vl_mask)
        return x[:, 0]

    def decode_from_rl_token(
        self,
        rl_token: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_mask: torch.Tensor,
        decoder_prefix_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct solely from learned positional queries and the RL token."""
        batch_size, seq_len, _ = target_embeddings.shape
        if seq_len > self.config.max_vl_tokens:
            raise ValueError(f"Expected at most {self.config.max_vl_tokens} tokens, got {seq_len}.")
        if tuple(target_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                f"Expected target_mask shape {(batch_size, seq_len)}, got {tuple(target_mask.shape)}."
            )
        if decoder_prefix_embeddings is not None:
            raise ValueError(
                "Strict openpi-RLT decoder does not accept ground-truth prefix embeddings."
            )

        x = self.decoder_query[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        memory = rl_token[:, None, :] + self.decoder_memory_pos.unsqueeze(0)
        for block in self.decoder:
            x = block(x, memory)
        return self.output_proj(x)

    def forward(
        self,
        vl_embeddings: torch.Tensor,
        vl_mask: torch.Tensor,
        decoder_prefix_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rl_token = self.encode_rl_token(vl_embeddings, vl_mask)
        reconstruction = self.decode_from_rl_token(
            rl_token,
            vl_embeddings,
            vl_mask,
            decoder_prefix_embeddings=decoder_prefix_embeddings,
        )
        return {"rl_token": rl_token, "reconstruction": reconstruction}


def load_autoencoder_state_dict(
    autoencoder: VLTokenAutoencoder,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load checkpoints while allowing newly introduced optional prefix-mask state."""
    incompatible = autoencoder.load_state_dict(state_dict, strict=False)
    allowed_missing = {"prefix_mask_token"} if autoencoder.prefix_mask_token is not None else set()
    allowed_unexpected = {"prefix_mask_token"} if autoencoder.prefix_mask_token is None else set()
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    bad_missing = sorted(missing - allowed_missing)
    bad_unexpected = sorted(unexpected - allowed_unexpected)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Failed to load autoencoder checkpoint with incompatible keys: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    if missing:
        print(f"Initialized new autoencoder parameter(s): {sorted(missing)}")
    if unexpected:
        print(f"Ignored unused autoencoder checkpoint parameter(s): {sorted(unexpected)}")


_STRICT_STATE_SENTINELS = {
    "query_token",
    "encoder_memory_pos",
    "decoder_query",
    "decoder_memory_pos",
}


def validate_strict_checkpoint_payload(checkpoint: dict[str, Any]) -> None:
    """Reject checkpoints that do not implement the strict cross-attention model."""

    architecture = checkpoint.get("architecture")
    schema_version = checkpoint.get("schema_version")
    state = checkpoint.get("autoencoder")
    if not isinstance(state, dict):
        raise ValueError("RL-token checkpoint does not contain an autoencoder state dict")
    missing_sentinels = sorted(_STRICT_STATE_SENTINELS - set(state))
    if missing_sentinels:
        raise ValueError(
            "RL-token checkpoint is not the strict cross-attention architecture; "
            f"missing sentinel keys={missing_sentinels}"
        )
    if architecture is None and schema_version is None:
        # One-time compatibility for the immutable 2026-07-14 10k archive.
        return
    if architecture != RL_TOKEN_ARCHITECTURE:
        raise ValueError(
            f"RL-token checkpoint architecture={architecture!r}, expected {RL_TOKEN_ARCHITECTURE!r}"
        )
    if int(schema_version) != RL_TOKEN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"RL-token checkpoint schema_version={schema_version!r}, expected "
            f"{RL_TOKEN_CHECKPOINT_SCHEMA_VERSION}"
        )


def validate_cached_embedding_contract(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Require a direct checkpoint-prefix cache with no secondary token selection."""

    expected = {
        "schema_version": VL_CACHE_SCHEMA_VERSION,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": VL_CACHE_FEATURE_TAP,
        "processor_mode": VL_CACHE_PROCESSOR_MODE,
        "token_scope": args.token_scope,
        "token_sampling": args.token_sampling,
        "max_vl_tokens": int(args.max_vl_tokens),
        "episode_sampling_rate": float(args.episode_sampling_rate),
        "dataset_sampling_seed": int(args.seed),
    }
    mismatches = [
        f"{name}: cache={manifest.get(name)!r} expected={value!r}"
        for name, value in expected.items()
        if manifest.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "Embedding cache is not a direct serving-equivalent prefix cache: "
            + "; ".join(mismatches)
        )


def load_modality_config(path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Modality config path does not exist: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"Loaded modality config: {path}")


def resolve_path(path: str | Path) -> str:
    """Resolve local filesystem paths while leaving Hugging Face repo ids intact."""
    text = str(path)
    expanded = Path(text).expanduser()
    if expanded.exists() or text.startswith(("/", "./", "../", "~")):
        return str(expanded.resolve())
    return text


class VLOnlyLeRobotDataset(IterableDataset):
    """Iterable image-language dataset for VLA embedding reconstruction.

    It intentionally ignores state/action modalities. The LeRobot dataset is used
    only as a source of synchronized camera frames and task instruction text.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, Any],
        processor: Gr00tN1d7Processor,
        video_backend: str,
        episode_sampling_rate: float,
        seed: int,
        instruction: str | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.video_modality = modality_configs["video"]
        self.language_modality = modality_configs["language"]
        self.instruction = instruction
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.epoch = 0
        self.loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs={
                "video": self.video_modality,
                "language": self.language_modality,
            },
            video_backend=video_backend,
        )
        if self.video_modality.delta_indices != [0]:
            raise ValueError(
                "VL-only autoencoder currently expects video.delta_indices=[0], "
                f"got {self.video_modality.delta_indices}"
            )
        if self.language_modality.delta_indices != [0]:
            raise ValueError(
                "VL-only autoencoder currently expects language.delta_indices=[0], "
                f"got {self.language_modality.delta_indices}"
            )

    def _sample_step_indices(self, length: int, rng: np.random.Generator) -> np.ndarray:
        indices = np.arange(length)
        rng.shuffle(indices)
        if self.episode_sampling_rate < 1.0:
            keep = max(1, int(round(length * self.episode_sampling_rate)))
            indices = indices[:keep]
        return indices

    def iter_epoch(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        episode_indices = np.arange(len(self.loader))
        rng.shuffle(episode_indices)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # IterableDataset copies must own disjoint episodes across workers.
            episode_indices = episode_indices[worker_info.id :: worker_info.num_workers]
        for episode_index in episode_indices:
            episode = self.loader[int(episode_index)]
            step_indices = self._sample_step_indices(len(episode), rng)
            image_keys = self.video_modality.modality_keys
            language_key = self.language_modality.modality_keys[0]
            language_column = f"language.{language_key}"
            for step_index in step_indices:
                images = {
                    key: [episode[f"video.{key}"].iloc[int(step_index)]] for key in image_keys
                }
                language = self.instruction or str(episode[language_column].iloc[int(step_index)])
                if self.processor.formalize_language:
                    language = re.sub(r"[^\w\s]", "", language.lower())
                yield self.processor._get_vlm_inputs(
                    image_keys=image_keys,
                    images=images,
                    masks=None,
                    image_transform=self.processor.eval_image_transform,
                    language=language,
                )

    def __iter__(self):
        while True:
            yield from self.iter_epoch()
            self.epoch += 1


class OneEpochVLOnlyDataset(IterableDataset):
    """Finite wrapper used when materializing a frozen embedding cache."""

    def __init__(self, dataset: VLOnlyLeRobotDataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        yield from self.dataset.iter_epoch()


class CachedVLEmbeddingDataset(IterableDataset):
    """Infinite training stream over precomputed VL embedding shards."""

    def __init__(self, cache_dir: str | Path, seed: int) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Embedding cache manifest does not exist: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise RuntimeError(f"No embedding shards found in cache manifest: {manifest_path}")
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        while True:
            shard_indices = np.arange(len(self.shards))
            rng.shuffle(shard_indices)
            for shard_index in shard_indices:
                shard_info = self.shards[int(shard_index)]
                shard = torch.load(self.cache_dir / shard_info["file"], map_location="cpu")
                sample_indices = np.arange(int(shard["packed"].shape[0]))
                rng.shuffle(sample_indices)
                for sample_index in sample_indices:
                    idx = int(sample_index)
                    yield {
                        "packed": shard["packed"][idx].float(),
                        "packed_mask": shard["packed_mask"][idx].bool(),
                        "packed_image_mask": shard["packed_image_mask"][idx].bool(),
                        "token_count": int(shard["token_counts"][idx]),
                        "selected_count": int(shard["selected_counts"][idx]),
                    }
            self.epoch += 1


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groot-repo-path",
        type=str,
        default=str(REPO_ROOT),
        help="Isaac-GR00T checkout used for models, examples, data, and default paths.",
    )

    parser.add_argument("--dataset-dir", type=str, default=str(L10_PREPARED_DATASET_DIR))
    parser.add_argument("--embodiment-tag", type=str, default=EmbodimentTag.NEW_EMBODIMENT.value)
    parser.add_argument("--modality-config-path", type=str, default=str(L10_MODALITY_CONFIG_PATH))
    parser.add_argument("--base-model-path", type=str, default=str(L10_BASE_MODEL_PATH))
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help=(
            "Checkpoint processor directory. Defaults to the processor saved with "
            "--base-model-path."
        ),
    )
    parser.add_argument("--vlm-model-path", type=str, default=str(L10_VLM_MODEL_PATH))
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional instruction override. If omitted, language is read from the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "IsaacLab" / "vl_embedding_autoencoder"),
    )

    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Use -1 for every CPU visible to the process.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="Step at which cosine decay reaches --min-learning-rate. Defaults to --max-steps.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="Optional parameter EMA decay. Checkpoints expose EMA weights for inference.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--fail-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop before an optimizer update when loss or gradient norm is non-finite.",
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=str(VL_EMBEDDING_CACHE_DIR),
        help=(
            "Directory containing precomputed packed VL embedding shards. "
            "Defaults to the Groot-RLT project outputs/cache tree."
        ),
    )
    parser.add_argument(
        "--precompute-vl-embeddings",
        action="store_true",
        help="Materialize frozen VLM embeddings to --embedding-cache-dir and exit.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Allow replacing files in --embedding-cache-dir during precomputation.",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Storage dtype for cached embeddings.",
    )

    parser.add_argument("--token-scope", choices=("all", "image", "non_image"), default="all")
    parser.add_argument(
        "--token-sampling", choices=("head", "tail", "uniform", "random"), default="uniform"
    )
    parser.add_argument("--max-vl-tokens", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--rl-token-dim", type=int, default=2048)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--decoder-cross-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let decoder blocks cross-attend to z_rl as explicit memory, matching rlt-openpi.",
    )

    parser.add_argument(
        "--decoder-prefix-corruption",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Corrupt teacher-forced decoder prefix tokens during training and use a "
            "masked/unmasked weighted reconstruction loss."
        ),
    )
    parser.add_argument(
        "--prefix-mask-prob",
        type=float,
        default=0.3,
        help="Per-prefix-token probability of replacement by a learned mask token.",
    )
    parser.add_argument(
        "--prefix-span-mask-prob",
        type=float,
        default=0.0,
        help="Per-prefix-token probability of starting a learned-mask span.",
    )
    parser.add_argument("--prefix-span-mask-min-len", type=int, default=2)
    parser.add_argument("--prefix-span-mask-max-len", type=int, default=8)
    parser.add_argument(
        "--prefix-shuffle-prob",
        type=float,
        default=0.0,
        help="Per-prefix-token probability of replacing the token from another batch sample.",
    )
    parser.add_argument(
        "--prefix-noise-prob",
        type=float,
        default=0.0,
        help="Per-prefix-token probability of adding Gaussian embedding noise.",
    )
    parser.add_argument(
        "--prefix-noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise std for prefix tokens selected by --prefix-noise-prob.",
    )
    parser.add_argument(
        "--prefix-corruption-masked-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for output positions whose immediate prefix token was corrupted.",
    )
    parser.add_argument(
        "--prefix-corruption-unmasked-loss-weight",
        type=float,
        default=0.2,
        help="Loss weight for retained stable output positions.",
    )
    parser.add_argument(
        "--prefix-corruption-unmasked-keep-prob",
        type=float,
        default=0.25,
        help="Subsample probability for stable, uncorrupted output positions in the loss.",
    )

    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--num-shards-per-epoch", type=int, default=1000)
    parser.add_argument("--video-backend", type=str, default="torchcodec")
    parser.add_argument("--allow-padding", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--autoencoder-bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use bf16 autocast for autoencoder forward/backward on CUDA.",
    )
    parser.add_argument(
        "--use-flash-attention", action=argparse.BooleanOptionalAction, default=None
    )

    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="vl-embedding-autoencoder")
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-workspace", type=str, default=None)
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        default=None,
        help="Optional SwanLab mode, for example cloud, local, offline, or disabled.",
    )
    parser.add_argument("--swanlab-logdir", type=str, default=None)
    parser.add_argument(
        "--swanlab-tags",
        type=str,
        default="vl-autoencoder,rl-token,groot-n1d7",
        help="Comma-separated SwanLab tags.",
    )
    parser.add_argument(
        "--swanlab-log-steps",
        type=int,
        default=1,
        help="Log detailed SwanLab metrics every N optimizer steps when enabled.",
    )
    parser.add_argument(
        "--swanlab-log-model-steps",
        type=int,
        default=50,
        help="Log parameter and gradient norm diagnostics every N optimizer steps.",
    )
    parser.add_argument(
        "--swanlab-log-cache-steps",
        type=int,
        default=1,
        help="Log embedding-cache precompute progress every N saved shards.",
    )
    parser.add_argument(
        "--swanlab-eval-ablation-on-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run z_rl ablation evaluation and log it to SwanLab whenever a checkpoint is saved.",
    )
    parser.add_argument("--ablation-eval-batches", type=int, default=32)
    parser.add_argument("--ablation-batch-size", type=int, default=16)
    parser.add_argument("--ablation-noise-std", type=float, default=1.0)
    parser.add_argument("--ablation-dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--swanlab-log-per-dim-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log per-embedding-dimension MSE diagnostics. This can create many metrics.",
    )
    parser.add_argument(
        "--swanlab-max-per-dim-logs",
        type=int,
        default=64,
        help="Maximum embedding dimensions to log when --swanlab-log-per-dim-loss is enabled.",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_backbone(args: argparse.Namespace, device: torch.device):
    model_path = resolve_path(args.base_model_path)
    model_cfg = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if args.vlm_model_path:
        model_cfg.model_name = resolve_path(args.vlm_model_path)

    use_cuda = device.type == "cuda"
    load_bf16 = args.load_bf16 if args.load_bf16 is not None else bool(use_cuda)
    use_flash_attention = (
        args.use_flash_attention
        if args.use_flash_attention is not None
        else bool(use_cuda and getattr(model_cfg, "use_flash_attention", False))
    )

    transformers_loading_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if getattr(model_cfg, "model_revision", None) is not None:
        transformers_loading_kwargs["revision"] = model_cfg.model_revision

    model_cfg.use_flash_attention = use_flash_attention
    model_cfg.load_bf16 = load_bf16
    full_model = AutoModel.from_pretrained(
        model_path,
        config=model_cfg,
        transformers_loading_kwargs=transformers_loading_kwargs,
        local_files_only=args.local_files_only,
    )
    if not hasattr(full_model, "backbone"):
        raise RuntimeError(f"Checkpoint model at {model_path} does not expose model.backbone")
    backbone = full_model.backbone
    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(
        device=device,
        dtype=torch.bfloat16 if load_bf16 else torch.float32,
    )
    del full_model
    gc.collect()
    return backbone, model_cfg, transformers_loading_kwargs


def build_dataset_and_processor(
    args: argparse.Namespace,
    model_cfg: Any,
    transformers_loading_kwargs: dict[str, Any],
):
    model_path = Path(resolve_path(args.base_model_path))
    processor_path_arg = getattr(args, "processor_path", None)
    processor_path = (
        Path(resolve_path(processor_path_arg))
        if processor_path_arg is not None
        else resolve_processor_path(model_path)
    )
    processor = AutoProcessor.from_pretrained(
        processor_path,
        transformers_loading_kwargs=transformers_loading_kwargs,
        model_name=model_cfg.model_name,
    )
    processor.eval()

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    checkpoint_modality_configs = processor.get_modality_configs()
    if embodiment_tag.value not in checkpoint_modality_configs:
        raise ValueError(
            f"Checkpoint processor at {processor_path} does not contain embodiment "
            f"{embodiment_tag.value!r}"
        )
    checkpoint_full_config = checkpoint_modality_configs[embodiment_tag.value]
    vl_modality_config = {
        "video": checkpoint_full_config["video"],
        "language": checkpoint_full_config["language"],
    }

    # The checkpoint processor is authoritative.  An external modality file is
    # accepted only as an explicit compatibility assertion.
    load_modality_config(args.modality_config_path)
    configured_full = MODALITY_CONFIGS[embodiment_tag.value]
    for group in ("video", "language"):
        expected = vl_modality_config[group]
        configured = configured_full[group]
        expected_signature = (
            tuple(expected.modality_keys),
            tuple(expected.delta_indices),
        )
        configured_signature = (
            tuple(configured.modality_keys),
            tuple(configured.delta_indices),
        )
        if configured_signature != expected_signature:
            raise ValueError(
                f"External modality config {group}={configured_signature!r} does not match "
                f"checkpoint processor {expected_signature!r}"
            )

    train_dataset = VLOnlyLeRobotDataset(
        dataset_path=resolve_path(args.dataset_dir),
        modality_configs=vl_modality_config,
        processor=processor,
        video_backend=args.video_backend,
        episode_sampling_rate=args.episode_sampling_rate,
        seed=args.seed,
        instruction=args.instruction,
    )
    return train_dataset, processor


def move_backbone_inputs(
    inputs: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> BatchFeature:
    keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    moved = {}
    for key in keys_to_use:
        value = inputs[key]
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return BatchFeature(moved)


def subsample_indices(
    indices: torch.Tensor,
    max_tokens: int,
    strategy: TokenSampling,
) -> torch.Tensor:
    if indices.numel() <= max_tokens:
        return indices
    if strategy == "head":
        return indices[:max_tokens]
    if strategy == "tail":
        return indices[-max_tokens:]
    if strategy == "random":
        selected = torch.randperm(indices.numel(), device=indices.device)[:max_tokens]
        selected = torch.sort(selected).values
        return indices[selected]

    positions = torch.linspace(
        0,
        indices.numel() - 1,
        steps=max_tokens,
        device=indices.device,
    ).round()
    return indices[positions.long()]


def pack_vl_tokens(
    backbone_output: BatchFeature,
    token_scope: TokenScope,
    max_tokens: int,
    token_sampling: TokenSampling,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    features = backbone_output["backbone_features"]
    attention_mask = backbone_output["backbone_attention_mask"].bool()
    image_mask = backbone_output["image_mask"].bool()

    if token_scope == "image":
        valid_mask = attention_mask & image_mask
    elif token_scope == "non_image":
        valid_mask = attention_mask & ~image_mask
    else:
        valid_mask = attention_mask

    batch_size, _, hidden_dim = features.shape
    packed = features.new_zeros((batch_size, max_tokens, hidden_dim))
    packed_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=features.device)
    packed_image_mask = torch.zeros_like(packed_mask)
    original_counts: list[int] = []
    selected_counts: list[int] = []

    for batch_idx in range(batch_size):
        indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        original_counts.append(int(indices.numel()))
        indices = subsample_indices(indices, max_tokens, token_sampling)
        selected_counts.append(int(indices.numel()))
        if indices.numel() == 0:
            continue
        count = int(indices.numel())
        packed[batch_idx, :count] = features[batch_idx, indices]
        packed_mask[batch_idx, :count] = True
        packed_image_mask[batch_idx, :count] = image_mask[batch_idx, indices]

    if not packed_mask.any():
        raise RuntimeError(
            f"No valid VL tokens found for token_scope={token_scope!r}. "
            "Try --token-scope all or inspect the VLA processor output."
        )
    return packed, packed_mask, packed_image_mask, original_counts, selected_counts


def compact_cached_vl_tokens(
    packed: torch.Tensor,
    packed_mask: torch.Tensor,
    packed_image_mask: torch.Tensor,
    *,
    token_scope: TokenScope,
    max_tokens: int,
    token_sampling: TokenSampling,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Select and compact cached tokens to the actual batch sequence length."""
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if packed.ndim != 3:
        raise ValueError(f"Expected packed shape [B, S, D], got {tuple(packed.shape)}")
    expected_mask_shape = packed.shape[:2]
    if tuple(packed_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"Expected packed_mask shape {expected_mask_shape}, got {tuple(packed_mask.shape)}"
        )
    if tuple(packed_image_mask.shape) != expected_mask_shape:
        raise ValueError(
            "Expected packed_image_mask shape "
            f"{expected_mask_shape}, got {tuple(packed_image_mask.shape)}"
        )

    if token_scope == "image":
        valid_mask = packed_mask & packed_image_mask
    elif token_scope == "non_image":
        valid_mask = packed_mask & ~packed_image_mask
    else:
        valid_mask = packed_mask

    selected_indices: list[torch.Tensor] = []
    original_counts: list[int] = []
    selected_counts: list[int] = []
    for batch_idx in range(packed.shape[0]):
        indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        original_counts.append(int(indices.numel()))
        indices = subsample_indices(indices, max_tokens, token_sampling)
        selected_indices.append(indices)
        selected_counts.append(int(indices.numel()))

    compact_len = max(selected_counts, default=0)
    if compact_len == 0:
        raise RuntimeError(
            f"No cached VL tokens found for token_scope={token_scope!r}. "
            "Use a cache containing the requested token type."
        )

    batch_size, _, hidden_dim = packed.shape
    compact = packed.new_zeros((batch_size, compact_len, hidden_dim))
    compact_mask = torch.zeros(
        batch_size,
        compact_len,
        dtype=torch.bool,
        device=packed.device,
    )
    compact_image_mask = torch.zeros_like(compact_mask)
    for batch_idx, indices in enumerate(selected_indices):
        count = int(indices.numel())
        if count == 0:
            continue
        compact[batch_idx, :count] = packed[batch_idx, indices]
        compact_mask[batch_idx, :count] = True
        compact_image_mask[batch_idx, :count] = packed_image_mask[batch_idx, indices]

    return compact, compact_mask, compact_image_mask, original_counts, selected_counts


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=prediction.dtype)
    loss = (prediction - target).pow(2) * mask_f
    denom = mask_f.sum().clamp_min(1.0) * prediction.shape[-1]
    return loss.sum() / denom


def validate_prefix_corruption_args(args: argparse.Namespace) -> None:
    if args.decoder_prefix_corruption:
        raise ValueError(
            "The strict openpi-RLT decoder only receives learned position queries and z_rl; "
            "--decoder-prefix-corruption is incompatible because no ground-truth decoder "
            "prefix is consumed."
        )
    probability_args = {
        "prefix_mask_prob": args.prefix_mask_prob,
        "prefix_span_mask_prob": args.prefix_span_mask_prob,
        "prefix_shuffle_prob": args.prefix_shuffle_prob,
        "prefix_noise_prob": args.prefix_noise_prob,
        "prefix_corruption_unmasked_keep_prob": args.prefix_corruption_unmasked_keep_prob,
    }
    for name, value in probability_args.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1], got {value}.")
    if args.prefix_noise_std < 0:
        raise ValueError(f"--prefix-noise-std must be non-negative, got {args.prefix_noise_std}.")
    if args.prefix_span_mask_min_len < 1:
        raise ValueError(
            f"--prefix-span-mask-min-len must be >= 1, got {args.prefix_span_mask_min_len}."
        )
    if args.prefix_span_mask_max_len < args.prefix_span_mask_min_len:
        raise ValueError(
            "--prefix-span-mask-max-len must be >= --prefix-span-mask-min-len, "
            f"got {args.prefix_span_mask_max_len} < {args.prefix_span_mask_min_len}."
        )
    if args.prefix_corruption_masked_loss_weight < 0:
        raise ValueError(
            "--prefix-corruption-masked-loss-weight must be non-negative, "
            f"got {args.prefix_corruption_masked_loss_weight}."
        )
    if args.prefix_corruption_unmasked_loss_weight < 0:
        raise ValueError(
            "--prefix-corruption-unmasked-loss-weight must be non-negative, "
            f"got {args.prefix_corruption_unmasked_loss_weight}."
        )


def validate_optimizer_args(args: argparse.Namespace) -> None:
    if args.max_steps < 1:
        raise ValueError(f"--max-steps must be positive, got {args.max_steps}.")
    if args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be positive, got {args.learning_rate}.")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError(
            "--min-learning-rate must be between zero and --learning-rate, "
            f"got {args.min_learning_rate}."
        )
    if args.warmup_steps < 0:
        raise ValueError(f"--warmup-steps must be non-negative, got {args.warmup_steps}.")
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else args.max_steps
    if decay_steps <= args.warmup_steps:
        raise ValueError(
            "--lr-decay-steps must be greater than --warmup-steps, "
            f"got {decay_steps} <= {args.warmup_steps}."
        )
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError(
            "--adam-beta1 and --adam-beta2 must be in [0, 1), "
            f"got ({args.adam_beta1}, {args.adam_beta2})."
        )
    if args.adam_eps <= 0:
        raise ValueError(f"--adam-eps must be positive, got {args.adam_eps}.")
    if args.ema_decay is not None and not 0 <= args.ema_decay < 1:
        raise ValueError(f"--ema-decay must be in [0, 1), got {args.ema_decay}.")
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be non-negative, got {args.weight_decay}.")
    if args.grad_clip < 0:
        raise ValueError(f"--grad-clip must be non-negative, got {args.grad_clip}.")


def sample_prefix_span_mask(
    valid_prefix_mask: torch.Tensor,
    *,
    start_prob: float,
    min_len: int,
    max_len: int,
) -> torch.Tensor:
    span_mask = torch.zeros_like(valid_prefix_mask)
    if start_prob <= 0 or valid_prefix_mask.numel() == 0:
        return span_mask

    starts = (
        torch.rand(valid_prefix_mask.shape, device=valid_prefix_mask.device) < start_prob
    ) & valid_prefix_mask
    for batch_idx, start_idx in starts.nonzero(as_tuple=False).tolist():
        span_len = int(
            torch.randint(
                min_len,
                max_len + 1,
                (1,),
                device=valid_prefix_mask.device,
            ).item()
        )
        end_idx = min(start_idx + span_len, valid_prefix_mask.shape[1])
        span_mask[batch_idx, start_idx:end_idx] |= valid_prefix_mask[batch_idx, start_idx:end_idx]
    return span_mask


def make_decoder_prefix_corruption(
    *,
    autoencoder: VLTokenAutoencoder,
    target_embeddings: torch.Tensor,
    target_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor | int | float | None]:
    batch_size, seq_len, _ = target_embeddings.shape
    prediction_corruption_mask = torch.zeros_like(target_mask, dtype=torch.bool)
    stable_loss_mask = target_mask.detach().bool()
    if not args.decoder_prefix_corruption or seq_len <= 1:
        return {
            "decoder_prefix_embeddings": None,
            "prediction_corruption_mask": prediction_corruption_mask,
            "stable_loss_mask": stable_loss_mask,
            "mask_token_prefix_tokens": 0,
            "span_mask_prefix_tokens": 0,
            "shuffle_prefix_tokens": 0,
            "noise_prefix_tokens": 0,
        }
    if autoencoder.prefix_mask_token is None:
        raise RuntimeError(
            "Prefix corruption requires VLTokenAutoencoderConfig.use_prefix_mask_token=True."
        )

    prefix = target_embeddings[:, :-1]
    valid_prefix_mask = target_mask[:, :-1].detach().bool()
    corrupted_prefix = prefix.clone()

    token_mask = (
        torch.rand(valid_prefix_mask.shape, device=valid_prefix_mask.device) < args.prefix_mask_prob
    ) & valid_prefix_mask
    span_mask = sample_prefix_span_mask(
        valid_prefix_mask,
        start_prob=float(args.prefix_span_mask_prob),
        min_len=int(args.prefix_span_mask_min_len),
        max_len=int(args.prefix_span_mask_max_len),
    )
    learned_mask_positions = token_mask | span_mask
    if bool(learned_mask_positions.any()):
        mask_token = autoencoder.prefix_mask_token.to(dtype=corrupted_prefix.dtype)
        corrupted_prefix = torch.where(
            learned_mask_positions.unsqueeze(-1),
            mask_token.expand(batch_size, seq_len - 1, -1),
            corrupted_prefix,
        )

    remaining_prefix_mask = valid_prefix_mask & ~learned_mask_positions
    shuffle_mask = torch.zeros_like(valid_prefix_mask)
    if args.prefix_shuffle_prob > 0 and batch_size > 1:
        shuffle_mask = (
            torch.rand(valid_prefix_mask.shape, device=valid_prefix_mask.device)
            < args.prefix_shuffle_prob
        ) & remaining_prefix_mask
        if bool(shuffle_mask.any()):
            perm = torch.randperm(batch_size, device=target_embeddings.device)
            if bool(torch.all(perm == torch.arange(batch_size, device=target_embeddings.device))):
                perm = torch.roll(perm, shifts=1)
            shuffled_prefix = prefix[perm]
            corrupted_prefix = torch.where(
                shuffle_mask.unsqueeze(-1),
                shuffled_prefix,
                corrupted_prefix,
            )
            remaining_prefix_mask = remaining_prefix_mask & ~shuffle_mask

    noise_mask = torch.zeros_like(valid_prefix_mask)
    if args.prefix_noise_prob > 0 and args.prefix_noise_std > 0:
        noise_mask = (
            torch.rand(valid_prefix_mask.shape, device=valid_prefix_mask.device)
            < args.prefix_noise_prob
        ) & remaining_prefix_mask
        if bool(noise_mask.any()):
            noise = torch.randn_like(corrupted_prefix) * float(args.prefix_noise_std)
            corrupted_prefix = torch.where(
                noise_mask.unsqueeze(-1),
                corrupted_prefix + noise,
                corrupted_prefix,
            )

    prefix_corruption_mask = learned_mask_positions | shuffle_mask | noise_mask
    # With shifted teacher forcing, corrupting prefix token t[i] directly affects target t[i + 1].
    prediction_corruption_mask[:, 1:] = prefix_corruption_mask

    stable_loss_mask = target_mask.detach().bool() & ~prediction_corruption_mask
    keep_prob = float(args.prefix_corruption_unmasked_keep_prob)
    if keep_prob < 1.0:
        stable_keep = (
            torch.rand(stable_loss_mask.shape, device=stable_loss_mask.device) < keep_prob
        ) & stable_loss_mask
        stable_loss_mask = stable_keep

    return {
        "decoder_prefix_embeddings": corrupted_prefix,
        "prediction_corruption_mask": prediction_corruption_mask,
        "stable_loss_mask": stable_loss_mask,
        "mask_token_prefix_tokens": int(token_mask.sum().detach().cpu()),
        "span_mask_prefix_tokens": int(span_mask.sum().detach().cpu()),
        "shuffle_prefix_tokens": int(shuffle_mask.sum().detach().cpu()),
        "noise_prefix_tokens": int(noise_mask.sum().detach().cpu()),
    }


def prefix_corruption_reconstruction_loss(
    *,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    prediction_corruption_mask: torch.Tensor,
    stable_loss_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    corrupted_loss_mask = target_mask & prediction_corruption_mask
    stable_loss_mask = target_mask & stable_loss_mask
    corrupted_tokens = int(corrupted_loss_mask.sum().detach().cpu())
    stable_tokens = int(stable_loss_mask.sum().detach().cpu())

    corrupted_loss = masked_mse_loss(reconstruction, target, corrupted_loss_mask)
    stable_loss = masked_mse_loss(reconstruction, target, stable_loss_mask)
    if corrupted_tokens == 0:
        loss = stable_loss
    elif stable_tokens == 0:
        loss = float(args.prefix_corruption_masked_loss_weight) * corrupted_loss
    else:
        loss = (
            float(args.prefix_corruption_masked_loss_weight) * corrupted_loss
            + float(args.prefix_corruption_unmasked_loss_weight) * stable_loss
        )

    valid_tokens = int(target_mask.sum().detach().cpu())
    return loss, {
        "loss/reconstruction_corrupted_prefix_targets": float(corrupted_loss.detach().cpu()),
        "loss/reconstruction_stable_targets": float(stable_loss.detach().cpu()),
        "prefix_corruption/enabled": 1,
        "prefix_corruption/corrupted_target_tokens": corrupted_tokens,
        "prefix_corruption/stable_loss_tokens": stable_tokens,
        "prefix_corruption/corrupted_target_fraction": float(
            corrupted_tokens / max(valid_tokens, 1)
        ),
        "prefix_corruption/stable_loss_fraction": float(stable_tokens / max(valid_tokens, 1)),
        "prefix_corruption/masked_loss_weight": float(args.prefix_corruption_masked_loss_weight),
        "prefix_corruption/unmasked_loss_weight": float(
            args.prefix_corruption_unmasked_loss_weight
        ),
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0e-6)


def tensor_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().float()
    if values.numel() == 0:
        return {}
    return {
        f"{prefix}/mean": float(values.mean().cpu()),
        f"{prefix}/std": float(values.std(unbiased=False).cpu()),
        f"{prefix}/min": float(values.min().cpu()),
        f"{prefix}/max": float(values.max().cpu()),
        f"{prefix}/l2_mean": float(values.norm(dim=-1).mean().cpu()) if values.ndim >= 2 else 0.0,
    }


def reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    if not bool(mask.any()):
        return {}

    prediction = prediction.detach().float()
    target = target.detach().float()
    mask = mask.detach().bool()
    error = prediction - target
    token_mse = error.pow(2).mean(dim=-1)
    token_mae = error.abs().mean(dim=-1)
    token_l2 = error.pow(2).sum(dim=-1).sqrt()
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1.0e-8)
    target_l2 = target.pow(2).sum(dim=-1).sqrt()
    pred_l2 = prediction.pow(2).sum(dim=-1).sqrt()

    mask_f = mask.float()
    sq_error_sum = (error.pow(2) * mask_f.unsqueeze(-1)).sum()
    target_sq_sum = (target.pow(2) * mask_f.unsqueeze(-1)).sum().clamp_min(1.0e-6)
    mse = masked_mean(token_mse, mask)
    mae = masked_mean(token_mae, mask)

    return {
        f"{prefix}/mse": float(mse.cpu()),
        f"{prefix}/rmse": float(mse.sqrt().cpu()),
        f"{prefix}/mae": float(mae.cpu()),
        f"{prefix}/error_l2": float(masked_mean(token_l2, mask).cpu()),
        f"{prefix}/cosine_similarity": float(masked_mean(cosine, mask).cpu()),
        f"{prefix}/cosine_distance": float((1.0 - masked_mean(cosine, mask)).cpu()),
        f"{prefix}/target_l2": float(masked_mean(target_l2, mask).cpu()),
        f"{prefix}/prediction_l2": float(masked_mean(pred_l2, mask).cpu()),
        f"{prefix}/relative_mse": float((sq_error_sum / target_sq_sum).cpu()),
    }


def per_dim_mse_logs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    max_dims: int,
) -> dict[str, float]:
    if max_dims <= 0 or not bool(mask.any()):
        return {}
    error_sq = (prediction.detach().float() - target.detach().float()).pow(2)
    mask_f = mask.detach().float().unsqueeze(-1)
    per_dim = (error_sq * mask_f).sum(dim=(0, 1)) / mask_f.sum().clamp_min(1.0e-6)
    logs: dict[str, float] = {}
    for dim in range(min(int(per_dim.numel()), max_dims)):
        logs[f"loss/per_dim_mse/dim_{dim:04d}"] = float(per_dim[dim].cpu())
    logs["loss/per_dim_mse/max"] = float(per_dim.max().cpu())
    logs["loss/per_dim_mse/mean"] = float(per_dim.mean().cpu())
    logs["loss/per_dim_mse/std"] = float(per_dim.std(unbiased=False).cpu())
    return logs


def grad_global_norm(parameters) -> float:
    total = torch.zeros((), dtype=torch.float32)
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total = total.to(grad.device) + grad.pow(2).sum()
    return float(total.sqrt().cpu())


def model_parameter_stats(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters/autoencoder_total": int(total),
        "parameters/autoencoder_trainable": int(trainable),
        "parameters/autoencoder_trainable_fraction": float(trainable / max(total, 1)),
    }


def parameter_group_name(name: str) -> str:
    if name == "query_token":
        return "query_token"
    if name.startswith("encoder."):
        parts = name.split(".")
        return f"encoder/layer_{parts[1]}" if len(parts) > 1 and parts[1].isdigit() else "encoder"
    if name.startswith("decoder."):
        parts = name.split(".")
        return f"decoder/layer_{parts[1]}" if len(parts) > 1 and parts[1].isdigit() else "decoder"
    if name.startswith("encoder_norm"):
        return "encoder_norm"
    if name.startswith("decoder_norm"):
        return "decoder_norm"
    if name.startswith("output_proj"):
        return "output_proj"
    return "other"


def model_norm_metrics(model: nn.Module) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    groups: dict[str, dict[str, float | int]] = {}
    total_param_sq = 0.0
    total_grad_sq = 0.0
    total_params = 0
    trainable_params = 0
    params_with_grad = 0

    for name, param in model.named_parameters():
        group = parameter_group_name(name)
        state = groups.setdefault(
            group,
            {
                "params": 0,
                "trainable": 0,
                "param_sq": 0.0,
                "param_abs_sum": 0.0,
                "param_abs_max": 0.0,
                "grad_sq": 0.0,
                "grad_abs_sum": 0.0,
                "grad_abs_max": 0.0,
                "grad_params": 0,
            },
        )
        param_count = int(param.numel())
        param_data = param.detach().float()
        param_sq = float(param_data.pow(2).sum().cpu())
        param_abs_sum = float(param_data.abs().sum().cpu())
        param_abs_max = float(param_data.abs().max().cpu()) if param_count > 0 else 0.0

        total_params += param_count
        total_param_sq += param_sq
        state["params"] = int(state["params"]) + param_count
        state["param_sq"] = float(state["param_sq"]) + param_sq
        state["param_abs_sum"] = float(state["param_abs_sum"]) + param_abs_sum
        state["param_abs_max"] = max(float(state["param_abs_max"]), param_abs_max)
        if param.requires_grad:
            trainable_params += param_count
            state["trainable"] = int(state["trainable"]) + param_count

        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_sq = float(grad.pow(2).sum().cpu())
        grad_abs_sum = float(grad.abs().sum().cpu())
        grad_abs_max = float(grad.abs().max().cpu()) if param_count > 0 else 0.0
        total_grad_sq += grad_sq
        params_with_grad += param_count
        state["grad_sq"] = float(state["grad_sq"]) + grad_sq
        state["grad_abs_sum"] = float(state["grad_abs_sum"]) + grad_abs_sum
        state["grad_abs_max"] = max(float(state["grad_abs_max"]), grad_abs_max)
        state["grad_params"] = int(state["grad_params"]) + param_count

    metrics["parameters/total"] = int(total_params)
    metrics["parameters/trainable"] = int(trainable_params)
    metrics["parameters/l2_norm"] = float(total_param_sq**0.5)
    metrics["gradients/params_with_grad"] = int(params_with_grad)
    metrics["gradients/params_with_grad_fraction"] = float(params_with_grad / max(total_params, 1))
    metrics["gradients/l2_norm_pre_clip"] = float(total_grad_sq**0.5)

    for group, state in groups.items():
        params = int(state["params"])
        grad_params = int(state["grad_params"])
        safe_group = group.replace("/", "_")
        metrics[f"parameters/by_group/{safe_group}/count"] = params
        metrics[f"parameters/by_group/{safe_group}/l2_norm"] = float(
            float(state["param_sq"]) ** 0.5
        )
        metrics[f"parameters/by_group/{safe_group}/abs_mean"] = float(
            float(state["param_abs_sum"]) / max(params, 1)
        )
        metrics[f"parameters/by_group/{safe_group}/abs_max"] = float(state["param_abs_max"])
        metrics[f"gradients/by_group/{safe_group}/l2_norm_pre_clip"] = float(
            float(state["grad_sq"]) ** 0.5
        )
        metrics[f"gradients/by_group/{safe_group}/abs_mean"] = float(
            float(state["grad_abs_sum"]) / max(grad_params, 1)
        )
        metrics[f"gradients/by_group/{safe_group}/abs_max"] = float(state["grad_abs_max"])
        metrics[f"gradients/by_group/{safe_group}/params_with_grad_fraction"] = float(
            grad_params / max(params, 1)
        )

    return metrics


def cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "memory/cuda_allocated_mb": float(torch.cuda.memory_allocated(device) / 2**20),
        "memory/cuda_reserved_mb": float(torch.cuda.memory_reserved(device) / 2**20),
        "memory/cuda_max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "memory/cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 2**20),
    }


def build_swanlab_config(
    args: argparse.Namespace,
    ae_config: VLTokenAutoencoderConfig,
    model_cfg: Any,
    autoencoder: VLTokenAutoencoder,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "script": "groot-rlt-train-token",
        "args": vars(args),
        "autoencoder_config": asdict(ae_config),
        "backbone_model_name": model_cfg.model_name,
        "backbone_embedding_dim": int(
            getattr(model_cfg, "backbone_embedding_dim", ae_config.input_dim)
        ),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        **model_parameter_stats(autoencoder),
    }


def init_swanlab(
    args: argparse.Namespace,
    output_dir: Path,
    config: dict[str, Any],
):
    if not args.use_swanlab:
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("SwanLab logging requested, but swanlab is not installed.") from exc

    experiment_name = args.swanlab_experiment_name or output_dir.name
    tags = [tag.strip() for tag in args.swanlab_tags.split(",") if tag.strip()]
    init_kwargs = {
        "project": args.swanlab_project,
        "experiment_name": experiment_name,
        "config": config,
        "tags": tags,
    }
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    if args.swanlab_mode:
        init_kwargs["mode"] = args.swanlab_mode
    if args.swanlab_logdir:
        init_kwargs["logdir"] = args.swanlab_logdir

    if swanlab.get_run() is None:
        swanlab.init(**init_kwargs)
    else:
        swanlab.config.update(config)

    with (output_dir / "swanlab_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "project": args.swanlab_project,
                "workspace": args.swanlab_workspace,
                "mode": args.swanlab_mode,
                "logdir": args.swanlab_logdir,
                "run_id": experiment_name,
                "tags": tags,
            },
            f,
            indent=2,
        )
    return swanlab


def build_training_metrics(
    *,
    step: int,
    loss_value: float,
    lr: float,
    output: dict[str, torch.Tensor],
    packed: torch.Tensor,
    packed_mask: torch.Tensor,
    packed_image_mask: torch.Tensor,
    token_counts: list[int],
    selected_counts: list[int],
    grad_norm: float,
    step_seconds: float,
    args: argparse.Namespace,
    device: torch.device,
    prefix_corruption_metrics: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    reconstruction = output["reconstruction"]
    rl_token = output["rl_token"]
    valid_tokens = int(packed_mask.sum().detach().cpu())
    image_token_mask = packed_mask & packed_image_mask
    non_image_token_mask = packed_mask & ~packed_image_mask

    batch_size = int(packed.shape[0])
    padded_tokens = int(packed_mask.numel())
    original_count = float(sum(token_counts))
    selected_count = float(sum(selected_counts))
    image_selected = int(image_token_mask.sum().detach().cpu())
    non_image_selected = int(non_image_token_mask.sum().detach().cpu())
    token_counts_array = np.asarray(token_counts, dtype=np.float32)
    selected_counts_array = np.asarray(selected_counts, dtype=np.float32)
    metrics: dict[str, float | int] = {
        "train/step": int(step),
        "train/loss": float(loss_value),
        "train/lr": float(lr),
        "train/grad_global_norm": float(grad_norm),
        "time/step_seconds": float(step_seconds),
        "throughput/valid_tokens_per_second": float(valid_tokens / max(step_seconds, 1.0e-6)),
        "throughput/samples_per_second": float(batch_size / max(step_seconds, 1.0e-6)),
        "batch/size": batch_size,
        "batch/max_vl_tokens": int(packed.shape[1]),
        "batch/embedding_dim": int(packed.shape[2]),
        "tokens/valid": valid_tokens,
        "tokens/padded_total": padded_tokens,
        "tokens/padding_fraction": float(1.0 - valid_tokens / max(padded_tokens, 1)),
        "tokens/valid_per_sample": float(valid_tokens / max(batch_size, 1)),
        "tokens/original_min": int(min(token_counts)),
        "tokens/original_mean": float(np.mean(token_counts)),
        "tokens/original_std": float(np.std(token_counts_array)),
        "tokens/original_max": int(max(token_counts)),
        "tokens/selected_min": int(min(selected_counts)),
        "tokens/selected_mean": float(np.mean(selected_counts)),
        "tokens/selected_std": float(np.std(selected_counts_array)),
        "tokens/selected_max": int(max(selected_counts)),
        "tokens/truncation_fraction": float(
            max(original_count - selected_count, 0.0) / max(original_count, 1.0)
        ),
        "tokens/image_selected": image_selected,
        "tokens/non_image_selected": non_image_selected,
        "tokens/image_selected_fraction": float(image_selected / max(valid_tokens, 1)),
        "tokens/non_image_selected_fraction": float(non_image_selected / max(valid_tokens, 1)),
    }
    metrics.update(cuda_memory_metrics(device))
    if prefix_corruption_metrics is not None:
        metrics.update(prefix_corruption_metrics)
    else:
        metrics["prefix_corruption/enabled"] = int(bool(args.decoder_prefix_corruption))

    metrics.update(
        reconstruction_metrics(reconstruction, packed, packed_mask, "loss/reconstruction")
    )
    metrics.update(
        reconstruction_metrics(
            reconstruction, packed, image_token_mask, "loss/reconstruction_image"
        )
    )
    metrics.update(
        reconstruction_metrics(
            reconstruction,
            packed,
            non_image_token_mask,
            "loss/reconstruction_non_image",
        )
    )
    metrics.update(tensor_stats(rl_token, "rl_token"))
    metrics.update(tensor_stats(packed[packed_mask], "embedding/target"))
    metrics.update(tensor_stats(reconstruction[packed_mask], "embedding/reconstruction"))
    metrics.update(tensor_stats((reconstruction - packed)[packed_mask], "embedding/error"))

    metrics["embedding/target_global_mean"] = float(
        packed[packed_mask].detach().float().mean().cpu()
    )
    metrics["embedding/reconstruction_global_mean"] = float(
        reconstruction[packed_mask].detach().float().mean().cpu()
    )
    metrics["rl_token/norm_min"] = float(rl_token.detach().float().norm(dim=-1).min().cpu())
    metrics["rl_token/norm_max"] = float(rl_token.detach().float().norm(dim=-1).max().cpu())
    metrics["rl_token/norm_std"] = float(
        rl_token.detach().float().norm(dim=-1).std(unbiased=False).cpu()
    )
    if args.swanlab_log_per_dim_loss:
        metrics.update(
            per_dim_mse_logs(
                reconstruction,
                packed,
                packed_mask,
                max_dims=args.swanlab_max_per_dim_logs,
            )
        )
    return metrics


def make_zrl_ablation_variants(
    z_rl: torch.Tensor,
    noise_std: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    variants = {
        "normal": z_rl,
        "zero": torch.zeros_like(z_rl),
    }
    if z_rl.shape[0] > 1:
        perm = torch.randperm(z_rl.shape[0], device=z_rl.device, generator=generator)
        if bool(torch.all(perm == torch.arange(z_rl.shape[0], device=z_rl.device))):
            perm = torch.roll(perm, shifts=1)
        variants["batch_shuffle"] = z_rl[perm]
    else:
        variants["batch_shuffle"] = z_rl

    if noise_std > 0:
        variants["noise"] = (
            z_rl
            + torch.randn(
                z_rl.shape,
                device=z_rl.device,
                dtype=z_rl.dtype,
                generator=generator,
            )
            * noise_std
        )
    else:
        variants["noise"] = z_rl
    return variants


def run_zrl_ablation_eval(
    *,
    autoencoder: VLTokenAutoencoder,
    data_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    autoencoder_bf16: bool,
) -> dict[str, float | int]:
    was_training = autoencoder.training
    autoencoder.eval()
    total_tokens = 0
    loss_sums = {
        "normal": 0.0,
        "zero": 0.0,
        "batch_shuffle": 0.0,
        "noise": 0.0,
    }
    cosine_sums = {key: 0.0 for key in loss_sums}
    relative_mse_sums = {key: 0.0 for key in loss_sums}
    batches = 0
    tic = time.time()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 12_345)

    with torch.no_grad():
        for batch in data_loader:
            if batches >= args.ablation_eval_batches:
                break
            batches += 1
            packed, packed_mask, _, _, _ = compact_cached_vl_tokens(
                batch["packed"],
                batch["packed_mask"].bool(),
                batch["packed_image_mask"].bool(),
                token_scope=args.token_scope,
                max_tokens=args.max_vl_tokens,
                token_sampling=args.token_sampling,
            )
            packed = packed.to(device=device, dtype=torch.float32, non_blocking=True)
            packed_mask = packed_mask.to(device=device, non_blocking=True)
            valid_tokens = int(packed_mask.sum().detach().cpu())
            total_tokens += valid_tokens

            with autocast_context(device, autoencoder_bf16):
                z_rl = autoencoder.encode_rl_token(packed, packed_mask)
                variants = make_zrl_ablation_variants(
                    z_rl,
                    args.ablation_noise_std,
                    generator,
                )
                for name, variant_z_rl in variants.items():
                    reconstruction = autoencoder.decode_from_rl_token(
                        variant_z_rl,
                        packed,
                        packed_mask,
                    )
                    loss = masked_mse_loss(reconstruction, packed, packed_mask)
                    metrics = reconstruction_metrics(
                        reconstruction,
                        packed,
                        packed_mask,
                        f"ablation/{name}",
                    )
                    loss_sums[name] += float(loss.detach().cpu()) * valid_tokens
                    cosine_sums[name] += (
                        float(metrics[f"ablation/{name}/cosine_similarity"]) * valid_tokens
                    )
                    relative_mse_sums[name] += (
                        float(metrics[f"ablation/{name}/relative_mse"]) * valid_tokens
                    )

    if was_training:
        autoencoder.train()

    denom = max(total_tokens, 1)
    normal_loss = loss_sums["normal"] / denom
    logs: dict[str, float | int] = {
        "ablation_eval/batches": int(batches),
        "ablation_eval/valid_tokens": int(total_tokens),
        "ablation_eval/elapsed_seconds": float(time.time() - tic),
        "ablation_eval/noise_std": float(args.ablation_noise_std),
    }
    for name in ("normal", "zero", "batch_shuffle", "noise"):
        loss_value = loss_sums[name] / denom
        logs[f"ablation_eval/{name}/loss"] = float(loss_value)
        logs[f"ablation_eval/{name}/cosine_similarity"] = float(cosine_sums[name] / denom)
        logs[f"ablation_eval/{name}/relative_mse"] = float(relative_mse_sums[name] / denom)
        logs[f"ablation_eval/{name}/loss_delta_vs_normal"] = float(loss_value - normal_loss)
        logs[f"ablation_eval/{name}/loss_ratio_vs_normal"] = float(
            loss_value / normal_loss if normal_loss > 0 else float("nan")
        )
    return logs


def update_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> float:
    update_count = max(step - 1, 0)
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else args.max_steps
    initial_lr = args.learning_rate / (args.warmup_steps + 1)
    if args.warmup_steps > 0 and update_count < args.warmup_steps:
        progress = update_count / args.warmup_steps
        lr = initial_lr + (args.learning_rate - initial_lr) * progress
    else:
        progress = (update_count - args.warmup_steps) / max(
            1,
            decay_steps - args.warmup_steps,
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def make_ema_state(autoencoder: VLTokenAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in autoencoder.state_dict().items()}


@torch.no_grad()
def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    autoencoder: VLTokenAutoencoder,
    decay: float,
) -> None:
    current_state = autoencoder.state_dict()
    if ema_state.keys() != current_state.keys():
        raise RuntimeError("EMA state keys no longer match the autoencoder state.")
    for name, ema_value in ema_state.items():
        current_value = current_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(current_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(current_value)


def save_checkpoint(
    path: Path,
    step: int,
    autoencoder: VLTokenAutoencoder,
    ema_state: dict[str, torch.Tensor] | None,
    optimizer: torch.optim.Optimizer,
    ae_config: VLTokenAutoencoderConfig,
    args: argparse.Namespace,
    last_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_state = autoencoder.state_dict()
    payload = {
        "schema_version": RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
        "architecture": RL_TOKEN_ARCHITECTURE,
        "step": step,
        "autoencoder": ema_state if ema_state is not None else raw_state,
        "optimizer": optimizer.state_dict(),
        "autoencoder_config": asdict(ae_config),
        "args": vars(args),
        "last_loss": last_loss,
    }
    if ema_state is not None:
        payload["autoencoder_raw"] = raw_state
        payload["ema_decay"] = args.ema_decay
    torch.save(payload, path)


def checkpoint_path_for_step(output_dir: Path, step: int) -> Path:
    return output_dir / f"{step:06d}.pt"


def checkpoint_step_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"(?:checkpoint_step_)?(\d+)\.pt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def prune_old_checkpoints(output_dir: Path, keep: int = 3) -> None:
    checkpoint_paths = []
    for path in output_dir.glob("*.pt"):
        step = checkpoint_step_from_path(path)
        if step is not None:
            checkpoint_paths.append((step, path.stat().st_mtime, path))
            continue
        if path.name in {"latest.pt", "final.pt"}:
            path.unlink()

    checkpoint_paths.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, path in checkpoint_paths[max(keep, 0) :]:
        path.unlink()


def maybe_load_checkpoint(
    path: str | None,
    autoencoder: VLTokenAutoencoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict[str, torch.Tensor] | None]:
    if path is None:
        return 0, None
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    validate_strict_checkpoint_payload(ckpt)
    load_autoencoder_state_dict(
        autoencoder,
        ckpt.get("autoencoder_raw", ckpt["autoencoder"]),
    )
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as exc:
            print(
                f"Skipping optimizer state from checkpoint because the parameter set changed: {exc}"
            )
    ema_state = ckpt["autoencoder"] if "autoencoder_raw" in ckpt else None
    return int(ckpt.get("step", 0)), ema_state


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def resolve_dataloader_num_workers(value: int) -> int:
    if value == -1:
        return max(1, os.cpu_count() or 1)
    if value < 0:
        raise ValueError("--dataloader-num-workers must be -1 or a non-negative integer")
    return value


def flush_embedding_shard(
    *,
    cache_dir: Path,
    shard_id: int,
    packed_chunks: list[torch.Tensor],
    mask_chunks: list[torch.Tensor],
    image_mask_chunks: list[torch.Tensor],
    token_counts: list[int],
    selected_counts: list[int],
    storage_dtype: torch.dtype,
) -> dict[str, Any]:
    packed = torch.cat(packed_chunks, dim=0).to(dtype=storage_dtype, device="cpu")
    packed_mask = torch.cat(mask_chunks, dim=0).to(dtype=torch.bool, device="cpu")
    packed_image_mask = torch.cat(image_mask_chunks, dim=0).to(dtype=torch.bool, device="cpu")
    filename = f"shard_{shard_id:06d}.pt"
    shard_path = cache_dir / filename
    torch.save(
        {
            "packed": packed,
            "packed_mask": packed_mask,
            "packed_image_mask": packed_image_mask,
            "token_counts": torch.tensor(token_counts, dtype=torch.int32),
            "selected_counts": torch.tensor(selected_counts, dtype=torch.int32),
        },
        shard_path,
    )
    return {
        "file": filename,
        "sha256": file_sha256(shard_path),
        "num_samples": int(packed.shape[0]),
        "num_valid_tokens": int(packed_mask.sum().item()),
    }


def precompute_vl_embedding_cache(
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> None:
    if args.embedding_cache_dir is None:
        raise ValueError("--precompute-vl-embeddings requires --embedding-cache-dir")

    model_path = Path(resolve_path(args.base_model_path))
    model_fingerprint, model_file_hashes = checkpoint_fingerprint(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.embedding_cache_dir).expanduser().resolve()
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.overwrite_cache:
        raise FileExistsError(
            f"Embedding cache directory is not empty: {cache_dir}. "
            "Pass --overwrite-cache to replace it."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_cache:
        for shard_path in cache_dir.glob("shard_*.pt"):
            shard_path.unlink()
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

    print(f"Using device: {device}")
    backbone, model_cfg, transformers_loading_kwargs = build_backbone(args, device)
    train_dataset, processor = build_dataset_and_processor(
        args, model_cfg, transformers_loading_kwargs
    )
    dataloader_num_workers = resolve_dataloader_num_workers(args.dataloader_num_workers)
    data_loader = DataLoader(
        OneEpochVLOnlyDataset(train_dataset),
        batch_size=args.batch_size,
        collate_fn=processor.collator,
        num_workers=dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    storage_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.cache_dtype]
    shards: list[dict[str, Any]] = []
    packed_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    image_mask_chunks: list[torch.Tensor] = []
    token_counts_buffer: list[int] = []
    selected_counts_buffer: list[int] = []
    samples_in_buffer = 0
    total_samples = 0
    total_tokens = 0
    tic = time.time()
    swanlab_run = init_swanlab(
        args,
        output_dir,
        {
            "script": "groot-rlt-train-token",
            "phase": "precompute_vl_embeddings",
            "args": vars(args),
            "backbone_model_name": model_cfg.model_name,
            "backbone_embedding_dim": int(getattr(model_cfg, "backbone_embedding_dim", 2048)),
            "device": str(device),
            "cache_dir": str(cache_dir),
            "cache_dtype": args.cache_dtype,
            "batch_size": int(args.batch_size),
            "shard_size": int(args.shard_size),
        },
    )

    print(
        "Precomputing frozen VL embeddings: "
        f"cache_dir={cache_dir}, batch_size={args.batch_size}, shard_size={args.shard_size}, "
        f"dataloader_num_workers={dataloader_num_workers}"
    )
    for batch in data_loader:
        inputs = batch["inputs"]
        dtype = next(backbone.parameters()).dtype
        backbone_inputs = move_backbone_inputs(inputs, device, dtype)
        with torch.inference_mode(), autocast_context(device, dtype == torch.bfloat16):
            backbone_output = backbone(backbone_inputs)

        packed, packed_mask, packed_image_mask, token_counts, selected_counts = pack_vl_tokens(
            backbone_output,
            token_scope=args.token_scope,
            max_tokens=args.max_vl_tokens,
            token_sampling=args.token_sampling,
        )
        batch_samples = int(packed.shape[0])
        packed_chunks.append(packed.detach().cpu())
        mask_chunks.append(packed_mask.detach().cpu())
        image_mask_chunks.append(packed_image_mask.detach().cpu())
        token_counts_buffer.extend(token_counts)
        selected_counts_buffer.extend(selected_counts)
        samples_in_buffer += batch_samples
        total_samples += batch_samples
        total_tokens += int(packed_mask.sum().detach().cpu())

        if samples_in_buffer >= args.shard_size:
            shard_info = flush_embedding_shard(
                cache_dir=cache_dir,
                shard_id=len(shards),
                packed_chunks=packed_chunks,
                mask_chunks=mask_chunks,
                image_mask_chunks=image_mask_chunks,
                token_counts=token_counts_buffer,
                selected_counts=selected_counts_buffer,
                storage_dtype=storage_dtype,
            )
            shards.append(shard_info)
            elapsed = max(time.time() - tic, 1.0e-6)
            print(
                f"cached_samples={total_samples} cached_tokens={total_tokens} "
                f"shards={len(shards)} elapsed={elapsed:.1f}s"
            )
            if swanlab_run is not None and len(shards) % max(1, args.swanlab_log_cache_steps) == 0:
                swanlab_run.log(
                    {
                        "cache/shards": int(len(shards)),
                        "cache/samples": int(total_samples),
                        "cache/valid_tokens": int(total_tokens),
                        "cache/elapsed_seconds": float(elapsed),
                        "cache/samples_per_second": float(total_samples / elapsed),
                        "cache/valid_tokens_per_second": float(total_tokens / elapsed),
                        "cache/last_shard_samples": int(shard_info["num_samples"]),
                        "cache/last_shard_valid_tokens": int(shard_info["num_valid_tokens"]),
                        "cache/batch_size": int(args.batch_size),
                        "cache/shard_size": int(args.shard_size),
                        **cuda_memory_metrics(device),
                    },
                    step=int(len(shards)),
                )
            packed_chunks = []
            mask_chunks = []
            image_mask_chunks = []
            token_counts_buffer = []
            selected_counts_buffer = []
            samples_in_buffer = 0

    if samples_in_buffer > 0:
        shard_info = flush_embedding_shard(
            cache_dir=cache_dir,
            shard_id=len(shards),
            packed_chunks=packed_chunks,
            mask_chunks=mask_chunks,
            image_mask_chunks=image_mask_chunks,
            token_counts=token_counts_buffer,
            selected_counts=selected_counts_buffer,
            storage_dtype=storage_dtype,
        )
        shards.append(shard_info)

    manifest = {
        "schema_version": VL_CACHE_SCHEMA_VERSION,
        "representation_source": "groot_checkpoint_backbone",
        "feature_tap": VL_CACHE_FEATURE_TAP,
        "processor_mode": VL_CACHE_PROCESSOR_MODE,
        "checkpoint_fingerprint": model_fingerprint,
        "checkpoint_files": model_file_hashes,
        "dataset_dir": str(Path(resolve_path(args.dataset_dir))),
        "modality_config_path": str(Path(args.modality_config_path).expanduser().resolve()),
        "base_model_path": resolve_path(args.base_model_path),
        "processor_path": str(
            Path(resolve_path(getattr(args, "processor_path", None))).resolve()
            if getattr(args, "processor_path", None) is not None
            else resolve_processor_path(Path(resolve_path(args.base_model_path)))
        ),
        "vlm_model_path": resolve_path(args.vlm_model_path),
        "instruction": args.instruction,
        "video_modality_keys": list(train_dataset.video_modality.modality_keys),
        "language_modality_keys": list(train_dataset.language_modality.modality_keys),
        "video_backend": args.video_backend,
        "episode_sampling_rate": float(args.episode_sampling_rate),
        "dataset_sampling_seed": int(args.seed),
        "token_scope": args.token_scope,
        "token_sampling": args.token_sampling,
        "max_vl_tokens": int(args.max_vl_tokens),
        "input_dim": int(getattr(model_cfg, "backbone_embedding_dim", 2048)),
        "cache_dtype": args.cache_dtype,
        "num_samples": int(total_samples),
        "num_valid_tokens": int(total_tokens),
        "shards": shards,
    }
    manifest["fingerprint"] = canonical_json_sha256(manifest)
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "embedding_cache_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Finished precomputing {total_samples} samples / {total_tokens} valid tokens "
        f"into {len(shards)} shard(s): {cache_dir}"
    )
    if swanlab_run is not None:
        elapsed = max(time.time() - tic, 1.0e-6)
        swanlab_run.log(
            {
                "cache/final_shards": int(len(shards)),
                "cache/final_samples": int(total_samples),
                "cache/final_valid_tokens": int(total_tokens),
                "cache/final_elapsed_seconds": float(elapsed),
                "cache/final_samples_per_second": float(total_samples / elapsed),
                "cache/final_valid_tokens_per_second": float(total_tokens / elapsed),
                **cuda_memory_metrics(device),
            },
            step=int(len(shards)),
        )
        finish = getattr(swanlab_run, "finish", None)
        if callable(finish):
            finish()


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()
    validate_prefix_corruption_args(args)
    validate_optimizer_args(args)
    seed_everything(args.seed)

    device = get_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.precompute_vl_embeddings:
        precompute_vl_embedding_cache(args, device, output_dir)
        return

    using_cache = args.embedding_cache_dir is not None
    dataloader_num_workers = resolve_dataloader_num_workers(args.dataloader_num_workers)
    cache_manifest = None
    ablation_loader = None
    if using_cache:
        cache_dir = Path(args.embedding_cache_dir).expanduser().resolve()
        with (cache_dir / "manifest.json").open("r", encoding="utf-8") as f:
            cache_manifest = json.load(f)
        validate_cached_embedding_contract(cache_manifest, args)
        recorded_cache_fingerprint = cache_manifest.get("fingerprint")
        cache_material = {
            key: value for key, value in cache_manifest.items() if key != "fingerprint"
        }
        actual_cache_fingerprint = canonical_json_sha256(cache_material)
        if recorded_cache_fingerprint != actual_cache_fingerprint:
            raise ValueError(
                "Embedding cache manifest fingerprint is invalid: "
                f"recorded={recorded_cache_fingerprint!r} actual={actual_cache_fingerprint!r}"
            )
        for shard in cache_manifest.get("shards", []):
            shard_path = cache_dir / shard["file"]
            actual_shard_hash = file_sha256(shard_path)
            if shard.get("sha256") != actual_shard_hash:
                raise ValueError(
                    f"Embedding cache shard hash mismatch for {shard_path}: "
                    f"recorded={shard.get('sha256')!r} actual={actual_shard_hash!r}"
                )
        expected_checkpoint_fingerprint, _ = checkpoint_fingerprint(args.base_model_path)
        if cache_manifest.get("checkpoint_fingerprint") != expected_checkpoint_fingerprint:
            raise ValueError(
                "Embedding cache checkpoint fingerprint does not match --base-model-path: "
                f"cache={cache_manifest.get('checkpoint_fingerprint')!r} "
                f"current={expected_checkpoint_fingerprint!r}"
            )
        args.representation_lineage = {
            "cache_fingerprint": actual_cache_fingerprint,
            "checkpoint_fingerprint": expected_checkpoint_fingerprint,
            "feature_tap": VL_CACHE_FEATURE_TAP,
            "processor_mode": VL_CACHE_PROCESSOR_MODE,
        }
        model_cfg = argparse.Namespace(
            model_name=cache_manifest.get("vlm_model_path", "cached-vl-embeddings"),
            backbone_embedding_dim=int(cache_manifest.get("input_dim", 2048)),
        )
        backbone = None
        train_dataset = CachedVLEmbeddingDataset(cache_dir, seed=args.seed)
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=(device.type == "cuda"),
        )
        ablation_dataset = CachedVLEmbeddingDataset(cache_dir, seed=args.seed + 100_000)
        ablation_loader = DataLoader(
            ablation_dataset,
            batch_size=args.ablation_batch_size,
            num_workers=args.ablation_dataloader_num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Using device: {device}")
        print(f"Training from precomputed VL embedding cache: {cache_dir}")
    else:
        print(f"Using device: {device}")
        backbone, model_cfg, transformers_loading_kwargs = build_backbone(args, device)
        train_dataset, processor = build_dataset_and_processor(
            args, model_cfg, transformers_loading_kwargs
        )
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=processor.collator,
            num_workers=dataloader_num_workers,
            pin_memory=(device.type == "cuda"),
        )

    input_dim = int(getattr(model_cfg, "backbone_embedding_dim", 2048))
    ae_config = VLTokenAutoencoderConfig(
        input_dim=input_dim,
        model_dim=args.model_dim,
        rl_token_dim=args.rl_token_dim,
        max_vl_tokens=args.max_vl_tokens,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_prefix_mask_token=bool(args.decoder_prefix_corruption),
        use_decoder_cross_attention=bool(args.decoder_cross_attention),
    )
    autoencoder = VLTokenAutoencoder(ae_config).to(device=device, dtype=torch.float32)
    autoencoder_bf16 = (
        args.autoencoder_bf16 if args.autoencoder_bf16 is not None else device.type == "cuda"
    )
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    start_step, restored_ema_state = maybe_load_checkpoint(
        args.resume,
        autoencoder,
        optimizer,
        device,
    )
    ema_state = None
    if args.ema_decay is not None:
        ema_state = restored_ema_state or make_ema_state(autoencoder)

    tracking_config = build_swanlab_config(args, ae_config, model_cfg, autoencoder, device)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": RL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
                "architecture": RL_TOKEN_ARCHITECTURE,
                "args": vars(args),
                "autoencoder_config": asdict(ae_config),
                "backbone_model_name": model_cfg.model_name,
                "embedding_cache_manifest": cache_manifest,
                **model_parameter_stats(autoencoder),
            },
            f,
            indent=2,
        )
    swanlab_run = init_swanlab(args, output_dir, tracking_config)
    if swanlab_run is not None:
        swanlab_run.log(
            {
                **model_parameter_stats(autoencoder),
                "train/start_step": int(start_step),
                "train/max_steps": int(args.max_steps),
                "train/using_embedding_cache": int(using_cache),
                "train/autoencoder_bf16": int(bool(autoencoder_bf16)),
                "optimizer/lr_initial": float(args.learning_rate),
                "optimizer/lr_min": float(args.min_learning_rate),
                "optimizer/lr_decay_steps": int(
                    args.lr_decay_steps if args.lr_decay_steps is not None else args.max_steps
                ),
                "optimizer/adam_beta1": float(args.adam_beta1),
                "optimizer/adam_beta2": float(args.adam_beta2),
                "optimizer/adam_eps": float(args.adam_eps),
                "optimizer/ema_decay": (
                    float(args.ema_decay) if args.ema_decay is not None else 0.0
                ),
                "optimizer/weight_decay": float(args.weight_decay),
                "optimizer/grad_clip_norm": float(args.grad_clip),
                "data/cache_num_samples": int(cache_manifest.get("num_samples", 0))
                if cache_manifest
                else 0,
                "data/cache_num_valid_tokens": int(cache_manifest.get("num_valid_tokens", 0))
                if cache_manifest
                else 0,
                "data/cache_num_shards": int(len(cache_manifest.get("shards", [])))
                if cache_manifest
                else 0,
                "ablation_eval/enabled": int(
                    bool(args.swanlab_eval_ablation_on_checkpoint and ablation_loader is not None)
                ),
                "ablation_eval/batch_size": int(args.ablation_batch_size),
                "ablation_eval/max_batches": int(args.ablation_eval_batches),
                "ablation_eval/noise_std": float(args.ablation_noise_std),
                "prefix_corruption/enabled": int(bool(args.decoder_prefix_corruption)),
                "prefix_corruption/mask_prob": float(args.prefix_mask_prob),
                "prefix_corruption/unmasked_loss_weight": float(
                    args.prefix_corruption_unmasked_loss_weight
                ),
            },
            step=int(start_step),
        )

    print(
        "Training VL autoencoder: "
        f"input_dim={ae_config.input_dim}, model_dim={ae_config.model_dim}, "
        f"rl_token_dim={ae_config.rl_token_dim}, max_vl_tokens={ae_config.max_vl_tokens}, "
        f"decoder_prefix_corruption={int(args.decoder_prefix_corruption)}, "
        f"decoder_cross_attention={int(args.decoder_cross_attention)}"
    )

    running_loss = 0.0
    running_tokens = 0.0
    tic = time.time()
    step = start_step

    for batch in data_loader:
        if step >= args.max_steps:
            break
        step += 1
        step_start = time.time()
        lr = update_learning_rate(optimizer, step, args)

        if using_cache:
            packed, packed_mask, packed_image_mask, token_counts, selected_counts = (
                compact_cached_vl_tokens(
                    batch["packed"],
                    batch["packed_mask"].bool(),
                    batch["packed_image_mask"].bool(),
                    token_scope=args.token_scope,
                    max_tokens=args.max_vl_tokens,
                    token_sampling=args.token_sampling,
                )
            )
            packed = packed.to(device=device, dtype=torch.float32, non_blocking=True)
            packed_mask = packed_mask.to(device=device, non_blocking=True)
            packed_image_mask = packed_image_mask.to(device=device, non_blocking=True)
        else:
            inputs = batch["inputs"]
            dtype = next(backbone.parameters()).dtype
            backbone_inputs = move_backbone_inputs(inputs, device, dtype)

            with torch.inference_mode(), autocast_context(device, dtype == torch.bfloat16):
                backbone_output = backbone(backbone_inputs)

            packed, packed_mask, packed_image_mask, token_counts, selected_counts = pack_vl_tokens(
                backbone_output,
                token_scope=args.token_scope,
                max_tokens=args.max_vl_tokens,
                token_sampling=args.token_sampling,
            )
            packed = packed.detach().to(dtype=torch.float32)
            packed_mask = packed_mask.detach()
            packed_image_mask = packed_image_mask.detach()

        autoencoder.train()
        with autocast_context(device, autoencoder_bf16):
            prefix_corruption = make_decoder_prefix_corruption(
                autoencoder=autoencoder,
                target_embeddings=packed,
                target_mask=packed_mask,
                args=args,
            )
            output = autoencoder(
                packed,
                packed_mask,
                decoder_prefix_embeddings=prefix_corruption["decoder_prefix_embeddings"],
            )
            if args.decoder_prefix_corruption:
                loss, prefix_corruption_metrics = prefix_corruption_reconstruction_loss(
                    reconstruction=output["reconstruction"],
                    target=packed,
                    target_mask=packed_mask,
                    prediction_corruption_mask=prefix_corruption["prediction_corruption_mask"],
                    stable_loss_mask=prefix_corruption["stable_loss_mask"],
                    args=args,
                )
                prefix_corruption_metrics.update(
                    {
                        "prefix_corruption/mask_token_prefix_tokens": int(
                            prefix_corruption["mask_token_prefix_tokens"]
                        ),
                        "prefix_corruption/span_mask_prefix_tokens": int(
                            prefix_corruption["span_mask_prefix_tokens"]
                        ),
                        "prefix_corruption/shuffle_prefix_tokens": int(
                            prefix_corruption["shuffle_prefix_tokens"]
                        ),
                        "prefix_corruption/noise_prefix_tokens": int(
                            prefix_corruption["noise_prefix_tokens"]
                        ),
                    }
                )
            else:
                loss = masked_mse_loss(output["reconstruction"], packed, packed_mask)
                prefix_corruption_metrics = None

        optimizer.zero_grad(set_to_none=True)
        if args.fail_on_nonfinite and not bool(torch.isfinite(loss).detach()):
            loss_value = float(loss.detach().cpu())
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss_value}")
        loss.backward()
        model_metrics = {}
        if (
            swanlab_run is not None
            and args.swanlab_log_model_steps > 0
            and step % args.swanlab_log_model_steps == 0
        ):
            model_metrics = model_norm_metrics(autoencoder)
        if args.grad_clip > 0:
            grad_norm_value = float(
                nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip).detach().cpu()
            )
        else:
            grad_norm_value = grad_global_norm(autoencoder.parameters())
        if args.fail_on_nonfinite and not math.isfinite(grad_norm_value):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm_value}")
        optimizer.step()
        if ema_state is not None:
            update_ema_state(ema_state, autoencoder, args.ema_decay)
        step_seconds = time.time() - step_start

        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        running_tokens += float(packed_mask.sum().detach().cpu())

        if swanlab_run is not None and step % max(1, args.swanlab_log_steps) == 0:
            swanlab_run.log(
                build_training_metrics(
                    step=step,
                    loss_value=loss_value,
                    lr=lr,
                    output=output,
                    packed=packed,
                    packed_mask=packed_mask,
                    packed_image_mask=packed_image_mask,
                    token_counts=token_counts,
                    selected_counts=selected_counts,
                    grad_norm=grad_norm_value,
                    step_seconds=step_seconds,
                    args=args,
                    device=device,
                    prefix_corruption_metrics=prefix_corruption_metrics,
                )
                | model_metrics,
                step=step,
            )

        if step % args.log_steps == 0:
            elapsed = max(time.time() - tic, 1e-6)
            avg_loss = running_loss / args.log_steps
            tokens_per_second = running_tokens / elapsed
            print(
                f"step={step:06d} loss={avg_loss:.6f} lr={lr:.3e} "
                f"tokens/s={tokens_per_second:.1f} "
                f"raw_tokens[min/mean/max]={min(token_counts)}/"
                f"{np.mean(token_counts):.1f}/{max(token_counts)}"
            )
            running_loss = 0.0
            running_tokens = 0.0
            tic = time.time()

        if args.save_steps > 0 and step % args.save_steps == 0:
            save_checkpoint(
                checkpoint_path_for_step(output_dir, step),
                step,
                autoencoder,
                ema_state,
                optimizer,
                ae_config,
                args,
                loss_value,
            )
            prune_old_checkpoints(output_dir)
            if swanlab_run is not None:
                ablation_metrics = {}
                if args.swanlab_eval_ablation_on_checkpoint and ablation_loader is not None:
                    ablation_metrics = run_zrl_ablation_eval(
                        autoencoder=autoencoder,
                        data_loader=ablation_loader,
                        args=args,
                        device=device,
                        autoencoder_bf16=autoencoder_bf16,
                    )
                swanlab_run.log(
                    {
                        "checkpoint/saved_step": int(step),
                        "checkpoint/latest_step": int(step),
                        **ablation_metrics,
                    },
                    step=step,
                )

    final_loss = loss_value if "loss_value" in locals() else float("nan")
    save_checkpoint(
        checkpoint_path_for_step(output_dir, step),
        step,
        autoencoder,
        ema_state,
        optimizer,
        ae_config,
        args,
        final_loss,
    )
    prune_old_checkpoints(output_dir)
    print(f"Finished at step={step}. Saved checkpoints to: {output_dir}")
    if swanlab_run is not None:
        swanlab_run.log(
            {
                "train/final_loss": float(final_loss),
                "train/final_step": int(step),
            },
            step=int(step),
        )
        finish = getattr(swanlab_run, "finish", None)
        if callable(finish):
            finish()


if __name__ == "__main__":
    main()
