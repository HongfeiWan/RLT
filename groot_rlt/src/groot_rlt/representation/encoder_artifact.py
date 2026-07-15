#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Export and load a strict, encoder-only EMA artifact for online RLT serving.

The training checkpoint contains the raw model, EMA model, decoder, and optimizer.
Serving only needs the EMA query encoder. This module creates a separately hashed
artifact that remains bound to the exact GR00T checkpoint and VL-prefix cache used
for representation training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from groot_rlt.integration.artifact_lineage import canonical_json_sha256, file_sha256

SOURCE_CHECKPOINT_SCHEMA_VERSION = 2
SOURCE_ARCHITECTURE = "openpi_rlt_strict_cross_attention_v1"
ENCODER_ARTIFACT_SCHEMA_VERSION = 1
ENCODER_ARTIFACT_KIND = "groot_rlt.encoder_ema"
ENCODER_ARCHITECTURE = f"{SOURCE_ARCHITECTURE}.encoder_only"
ENCODER_MANIFEST_SCHEMA_VERSION = 1
ENCODER_MANIFEST_KIND = "groot_rlt.encoder_ema.manifest"

FEATURE_TAP = "raw_backbone_pre_action_head"
PROCESSOR_MODE = "eval"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SOURCE_CONFIG_KEYS = {
    "input_dim",
    "model_dim",
    "rl_token_dim",
    "max_vl_tokens",
    "encoder_layers",
    "decoder_layers",
    "num_heads",
    "mlp_ratio",
    "dropout",
    "use_prefix_mask_token",
    "use_decoder_cross_attention",
}
_LINEAGE_KEYS = {
    "cache_fingerprint",
    "checkpoint_fingerprint",
    "feature_tap",
    "processor_mode",
}
_ARTIFACT_ROOT_KEYS = {"metadata", "encoder"}
_METADATA_KEYS = {
    "schema_version",
    "artifact_kind",
    "architecture",
    "encoder_config",
    "representation_lineage",
    "source_checkpoint",
    "encoder_state_sha256",
}
_SOURCE_METADATA_KEYS = {
    "schema_version",
    "architecture",
    "step",
    "ema_decay",
    "sha256",
    "size_bytes",
}
_MANIFEST_KEYS = {
    "schema_version",
    "manifest_kind",
    "artifact_filename",
    "artifact_sha256",
    "artifact_size_bytes",
    "metadata_sha256",
    "encoder_state_sha256",
    "source_checkpoint_sha256",
    "source_checkpoint_size_bytes",
    "size_ratio_vs_source",
    "representation_lineage",
}


@dataclass(frozen=True)
class StrictEncoderConfig:
    """Structure required to recreate the strict openpi-RLT query encoder."""

    input_dim: int
    model_dim: int
    rl_token_dim: int
    max_vl_tokens: int
    encoder_layers: int
    num_heads: int
    mlp_ratio: float
    dropout: float

    def validate(self) -> None:
        """Validate dimensions before allocating model parameters."""

        integer_fields = {
            "input_dim": self.input_dim,
            "model_dim": self.model_dim,
            "rl_token_dim": self.rl_token_dim,
            "max_vl_tokens": self.max_vl_tokens,
            "encoder_layers": self.encoder_layers,
            "num_heads": self.num_heads,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"encoder_config.{name} must be a positive integer, got {value!r}")
        if self.rl_token_dim != self.model_dim:
            raise ValueError(
                "encoder_config.rl_token_dim must equal model_dim for the strict query encoder"
            )
        if self.model_dim % self.num_heads != 0:
            raise ValueError("encoder_config.model_dim must be divisible by num_heads")
        if self.model_dim % 2 != 0:
            raise ValueError("encoder_config.model_dim must be even for sinusoidal initialization")
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("encoder_config.mlp_ratio must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("encoder_config.dropout must be in [0, 1)")


def _sinusoidal_position_embeddings(seq_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    return torch.cat(
        [
            torch.sin(position[:, None] * div_term),
            torch.cos(position[:, None] * div_term),
        ],
        dim=-1,
    )


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = value.shape
        return value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_proj(value))
        key = self._split_heads(self.k_proj(value))
        projected_value = self._split_heads(self.v_proj(value))
        output = F.scaled_dot_product_attention(
            query,
            key,
            projected_value,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(value.shape[0], value.shape[1], -1)
        return self.o_proj(output)


class _MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = value.shape
        return value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        value: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.q_proj(value))
        key = self._split_heads(self.k_proj(memory))
        projected_value = self._split_heads(self.v_proj(memory))
        attention_mask = None
        if memory_key_padding_mask is not None:
            attention_mask = torch.zeros(
                value.shape[0],
                1,
                value.shape[1],
                memory.shape[1],
                dtype=query.dtype,
                device=value.device,
            )
            attention_mask = attention_mask.masked_fill(
                memory_key_padding_mask[:, None, None, :], -torch.inf
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            projected_value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(value.shape[0], value.shape[1], -1)
        return self.o_proj(output)


class _OpenPiGeGLUMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.geglu_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.dropout(self.input_proj(value))
        content, gate = self.geglu_proj(value).chunk(2, dim=-1)
        return self.output_proj(content * F.gelu(gate, approximate="tanh"))


class _OpenPiCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.self_attn = _MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.cross_attn = _MultiHeadCrossAttention(dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.mlp = _OpenPiGeGLUMLP(dim, mlp_ratio, dropout)

    def forward(
        self,
        value: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = value + self.self_attn(self.self_attn_norm(value))
        value = value + self.cross_attn(
            self.cross_attn_norm(value),
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return value + self.mlp(self.mlp_norm(value))


class StrictRLTokenEncoder(nn.Module):
    """Encoder-only form of the strict openpi-RLT cross-attention model."""

    def __init__(self, config: StrictEncoderConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.input_proj = (
            nn.Linear(config.input_dim, config.model_dim)
            if config.input_dim != config.model_dim
            else nn.Identity()
        )
        self.query_token = nn.Parameter(_sinusoidal_position_embeddings(1, config.model_dim))
        self.encoder_memory_pos = nn.Parameter(
            _sinusoidal_position_embeddings(config.max_vl_tokens, config.model_dim)
        )
        self.encoder = nn.ModuleList(
            [
                _OpenPiCrossAttentionBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )

    def encode_rl_token(self, vl_embeddings: torch.Tensor, vl_mask: torch.Tensor) -> torch.Tensor:
        """Encode a frozen VLA prefix into one RL token per batch item."""

        if vl_embeddings.ndim != 3:
            raise ValueError(
                f"Expected vl_embeddings shape [B, S, D], got {tuple(vl_embeddings.shape)}"
            )
        batch_size, seq_len, input_dim = vl_embeddings.shape
        if input_dim != self.config.input_dim:
            raise ValueError(f"Expected input dimension {self.config.input_dim}, got {input_dim}")
        if vl_mask.dtype is not torch.bool:
            raise ValueError(f"Expected a boolean vl_mask, got dtype={vl_mask.dtype}")
        if tuple(vl_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                f"Expected vl_mask shape {(batch_size, seq_len)}, got {tuple(vl_mask.shape)}"
            )
        empty_samples = torch.nonzero(~vl_mask.any(dim=1), as_tuple=False).flatten()
        if empty_samples.numel():
            raise ValueError(
                "Every RL-token sample must contain at least one valid prefix token; "
                f"empty batch indices={empty_samples.tolist()}"
            )
        if seq_len > self.config.max_vl_tokens:
            raise ValueError(f"Expected at most {self.config.max_vl_tokens} tokens, got {seq_len}")

        value = self.query_token.expand(batch_size, -1, -1)
        memory = self.input_proj(vl_embeddings)
        memory = memory + self.encoder_memory_pos[:seq_len].unsqueeze(0)
        for block in self.encoder:
            value = block(value, memory, memory_key_padding_mask=~vl_mask)
        return value[:, 0]

    def forward(self, vl_embeddings: torch.Tensor, vl_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_rl_token(vl_embeddings, vl_mask)


@dataclass(frozen=True)
class LoadedEncoderArtifact:
    """A verified encoder and its immutable artifact metadata."""

    encoder: StrictRLTokenEncoder
    metadata: dict[str, Any]
    artifact_path: Path
    manifest_path: Path


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"{label} keys mismatch: missing={missing}, unexpected={unexpected}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must have form sha256:<64 lowercase hex>, got {value!r}")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _normalise_lineage(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    args = checkpoint.get("args")
    if isinstance(args, Mapping) and isinstance(args.get("representation_lineage"), Mapping):
        candidates.append(("args.representation_lineage", args["representation_lineage"]))
    for name in ("representation_lineage", "lineage"):
        value = checkpoint.get(name)
        if isinstance(value, Mapping):
            candidates.append((name, value))
    if not candidates:
        raise ValueError("schema2 checkpoint is missing authoritative args.representation_lineage")

    normalised = dict(candidates[0][1])
    _require_exact_keys(normalised, _LINEAGE_KEYS, candidates[0][0])
    baseline = canonical_json_sha256(normalised)
    for name, candidate in candidates[1:]:
        candidate_dict = dict(candidate)
        _require_exact_keys(candidate_dict, _LINEAGE_KEYS, name)
        if canonical_json_sha256(candidate_dict) != baseline:
            raise ValueError(
                "checkpoint contains conflicting representation lineage: "
                f"{candidates[0][0]} != {name}"
            )
    _require_sha256(normalised["checkpoint_fingerprint"], "checkpoint_fingerprint")
    _require_sha256(normalised["cache_fingerprint"], "cache_fingerprint")
    if normalised["feature_tap"] != FEATURE_TAP:
        raise ValueError(
            f"lineage.feature_tap={normalised['feature_tap']!r}, expected {FEATURE_TAP!r}"
        )
    if normalised["processor_mode"] != PROCESSOR_MODE:
        raise ValueError(
            f"lineage.processor_mode={normalised['processor_mode']!r}, expected {PROCESSOR_MODE!r}"
        )
    return normalised


def _encoder_config_from_source(source: Mapping[str, Any]) -> StrictEncoderConfig:
    config = source.get("autoencoder_config")
    if not isinstance(config, Mapping):
        raise ValueError("schema2 checkpoint is missing autoencoder_config")
    _require_exact_keys(config, _SOURCE_CONFIG_KEYS, "autoencoder_config")
    if config["use_prefix_mask_token"] is not False:
        raise ValueError("strict source checkpoint must not use a decoder prefix mask token")
    if config["use_decoder_cross_attention"] is not True:
        raise ValueError("strict source checkpoint must use decoder cross-attention")
    _require_int(config["decoder_layers"], "autoencoder_config.decoder_layers", minimum=1)
    encoder_config = StrictEncoderConfig(
        input_dim=config["input_dim"],
        model_dim=config["model_dim"],
        rl_token_dim=config["rl_token_dim"],
        max_vl_tokens=config["max_vl_tokens"],
        encoder_layers=config["encoder_layers"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        dropout=config["dropout"],
    )
    encoder_config.validate()
    return encoder_config


def _block_state_shapes(prefix: str, dim: int, mlp_ratio: float) -> dict[str, tuple[int, ...]]:
    hidden_dim = int(dim * mlp_ratio)
    result: dict[str, tuple[int, ...]] = {}
    for norm in ("self_attn_norm", "cross_attn_norm", "mlp_norm"):
        result[f"{prefix}.{norm}.weight"] = (dim,)
        result[f"{prefix}.{norm}.bias"] = (dim,)
    for attention in ("self_attn", "cross_attn"):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            result[f"{prefix}.{attention}.{projection}.weight"] = (dim, dim)
            result[f"{prefix}.{attention}.{projection}.bias"] = (dim,)
    result[f"{prefix}.mlp.input_proj.weight"] = (hidden_dim, dim)
    result[f"{prefix}.mlp.input_proj.bias"] = (hidden_dim,)
    result[f"{prefix}.mlp.geglu_proj.weight"] = (hidden_dim * 2, hidden_dim)
    result[f"{prefix}.mlp.geglu_proj.bias"] = (hidden_dim * 2,)
    result[f"{prefix}.mlp.output_proj.weight"] = (dim, hidden_dim)
    result[f"{prefix}.mlp.output_proj.bias"] = (dim,)
    return result


def _encoder_state_shapes(config: StrictEncoderConfig) -> dict[str, tuple[int, ...]]:
    result = {
        "query_token": (1, config.model_dim),
        "encoder_memory_pos": (config.max_vl_tokens, config.model_dim),
    }
    if config.input_dim != config.model_dim:
        result["input_proj.weight"] = (config.model_dim, config.input_dim)
        result["input_proj.bias"] = (config.model_dim,)
    for layer in range(config.encoder_layers):
        result.update(_block_state_shapes(f"encoder.{layer}", config.model_dim, config.mlp_ratio))
    return result


def _full_state_shapes(
    source_config: Mapping[str, Any],
    encoder_config: StrictEncoderConfig,
) -> dict[str, tuple[int, ...]]:
    result = _encoder_state_shapes(encoder_config)
    result["decoder_query"] = (encoder_config.max_vl_tokens, encoder_config.model_dim)
    result["decoder_memory_pos"] = (1, encoder_config.model_dim)
    for layer in range(int(source_config["decoder_layers"])):
        result.update(
            _block_state_shapes(
                f"decoder.{layer}", encoder_config.model_dim, encoder_config.mlp_ratio
            )
        )
    if encoder_config.input_dim != encoder_config.model_dim:
        result["output_proj.weight"] = (encoder_config.input_dim, encoder_config.model_dim)
        result["output_proj.bias"] = (encoder_config.input_dim,)
    return result


def _validate_tensor_state(
    state: Mapping[str, Any],
    expected_shapes: Mapping[str, tuple[int, ...]],
    label: str,
    *,
    check_finite: bool = True,
) -> None:
    _require_exact_keys(state, set(expected_shapes), label)
    for name, expected_shape in expected_shapes.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{label}.{name} is not a tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{label}.{name} shape={tuple(value.shape)}, expected={expected_shape}"
            )
        if value.dtype is not torch.float32:
            raise ValueError(f"{label}.{name} dtype={value.dtype}, expected=torch.float32")
        if check_finite and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label}.{name} contains non-finite values")


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and raw contiguous bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(memoryview(value.numpy()).cast("B"))
    return f"sha256:{digest.hexdigest()}"


def artifact_manifest_path(artifact_path: str | Path) -> Path:
    """Return the mandatory sidecar path for an encoder artifact."""

    path = Path(artifact_path).expanduser().resolve()
    return path.with_suffix(path.suffix + ".manifest.json")


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        # AdamW's scheduled learning rate is serialized as numpy.float64 by the
        # training loop. Allow only the NumPy constructors needed for that scalar
        # while retaining weights-only unpickling for the large source checkpoint.
        numpy_safe_globals = [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(numpy_safe_globals):
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except TypeError as exc:  # pragma: no cover - supported by the pinned torch version.
        raise RuntimeError(
            "encoder artifacts require torch.load(weights_only=True, mmap=True)"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary payload in {path}")
    return payload


def export_encoder_ema_artifact(
    source_checkpoint: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export only strict EMA encoder weights and write a hashed sidecar manifest."""

    source_path = Path(source_checkpoint).expanduser().resolve()
    artifact_path = Path(output_path).expanduser().resolve()
    manifest_path = artifact_manifest_path(artifact_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_path}")
    if source_path == artifact_path:
        raise ValueError("Source checkpoint and encoder artifact paths must differ")
    if not overwrite:
        existing = [path for path in (artifact_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite encoder artifact files: {existing}")

    source = _load_torch_payload(source_path)
    if source.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"source schema_version={source.get('schema_version')!r}, "
            f"expected {SOURCE_CHECKPOINT_SCHEMA_VERSION}"
        )
    if source.get("architecture") != SOURCE_ARCHITECTURE:
        raise ValueError(
            f"source architecture={source.get('architecture')!r}, expected {SOURCE_ARCHITECTURE!r}"
        )
    step = _require_int(source.get("step"), "source.step", minimum=1)
    ema_decay = source.get("ema_decay")
    if not isinstance(ema_decay, (int, float)) or isinstance(ema_decay, bool):
        raise ValueError(f"source.ema_decay must be numeric, got {ema_decay!r}")
    ema_decay = float(ema_decay)
    if not math.isfinite(ema_decay) or not 0 < ema_decay < 1:
        raise ValueError(f"source.ema_decay must be in (0, 1), got {ema_decay!r}")
    if not isinstance(source.get("autoencoder"), Mapping):
        raise ValueError("source checkpoint is missing EMA autoencoder weights")
    if not isinstance(source.get("autoencoder_raw"), Mapping):
        raise ValueError(
            "source checkpoint has no autoencoder_raw state; cannot prove autoencoder is EMA"
        )

    encoder_config = _encoder_config_from_source(source)
    source_config = source["autoencoder_config"]
    expected_full = _full_state_shapes(source_config, encoder_config)
    ema_state = source["autoencoder"]
    raw_state = source["autoencoder_raw"]
    # The full-state pass validates architecture without materializing every mmap-backed
    # decoder/optimizer page. Finiteness is checked on the exported EMA encoder below.
    _validate_tensor_state(
        ema_state,
        expected_full,
        "source.autoencoder",
        check_finite=False,
    )
    _validate_tensor_state(
        raw_state,
        expected_full,
        "source.autoencoder_raw",
        check_finite=False,
    )
    lineage = _normalise_lineage(source)

    expected_encoder = _encoder_state_shapes(encoder_config)
    encoder_state = {name: ema_state[name].detach().cpu().contiguous() for name in expected_encoder}
    _validate_tensor_state(encoder_state, expected_encoder, "artifact.encoder")
    state_sha256 = tensor_state_sha256(encoder_state)
    source_sha256 = f"sha256:{file_sha256(source_path)}"
    source_size = source_path.stat().st_size
    metadata = {
        "schema_version": ENCODER_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ENCODER_ARTIFACT_KIND,
        "architecture": ENCODER_ARCHITECTURE,
        "encoder_config": asdict(encoder_config),
        "representation_lineage": lineage,
        "source_checkpoint": {
            "schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
            "architecture": SOURCE_ARCHITECTURE,
            "step": step,
            "ema_decay": ema_decay,
            "sha256": source_sha256,
            "size_bytes": source_size,
        },
        "encoder_state_sha256": state_sha256,
    }
    payload = {"metadata": metadata, "encoder": encoder_state}

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_tmp: Path | None = None
    manifest_tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{artifact_path.name}.", suffix=".tmp", dir=artifact_path.parent, delete=False
        ) as stream:
            artifact_tmp = Path(stream.name)
        torch.save(payload, artifact_tmp)
        artifact_sha256 = f"sha256:{file_sha256(artifact_tmp)}"
        artifact_size = artifact_tmp.stat().st_size
        manifest = {
            "schema_version": ENCODER_MANIFEST_SCHEMA_VERSION,
            "manifest_kind": ENCODER_MANIFEST_KIND,
            "artifact_filename": artifact_path.name,
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": artifact_size,
            "metadata_sha256": canonical_json_sha256(metadata),
            "encoder_state_sha256": state_sha256,
            "source_checkpoint_sha256": source_sha256,
            "source_checkpoint_size_bytes": source_size,
            "size_ratio_vs_source": artifact_size / source_size,
            "representation_lineage": lineage,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=artifact_path.parent,
            delete=False,
        ) as stream:
            manifest_tmp = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(artifact_tmp, artifact_path)
        artifact_tmp = None
        os.replace(manifest_tmp, manifest_path)
        manifest_tmp = None
        return manifest
    finally:
        for temporary in (artifact_tmp, manifest_tmp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _read_and_validate_manifest(artifact_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Encoder artifact manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Encoder artifact manifest must be a JSON object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != ENCODER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported encoder artifact manifest schema_version")
    if manifest["manifest_kind"] != ENCODER_MANIFEST_KIND:
        raise ValueError("Unexpected encoder artifact manifest_kind")
    if manifest["artifact_filename"] != artifact_path.name:
        raise ValueError(
            f"manifest artifact_filename={manifest['artifact_filename']!r}, "
            f"actual={artifact_path.name!r}"
        )
    _require_sha256(manifest["artifact_sha256"], "manifest.artifact_sha256")
    _require_sha256(manifest["metadata_sha256"], "manifest.metadata_sha256")
    _require_sha256(manifest["encoder_state_sha256"], "manifest.encoder_state_sha256")
    _require_sha256(manifest["source_checkpoint_sha256"], "manifest.source_checkpoint_sha256")
    _require_int(manifest["artifact_size_bytes"], "manifest.artifact_size_bytes", minimum=1)
    _require_int(
        manifest["source_checkpoint_size_bytes"],
        "manifest.source_checkpoint_size_bytes",
        minimum=1,
    )
    if not isinstance(manifest["size_ratio_vs_source"], (int, float)):
        raise ValueError("manifest.size_ratio_vs_source must be numeric")
    lineage = manifest["representation_lineage"]
    if not isinstance(lineage, Mapping):
        raise ValueError("manifest.representation_lineage must be an object")
    _require_exact_keys(lineage, _LINEAGE_KEYS, "manifest.representation_lineage")
    if artifact_path.stat().st_size != manifest["artifact_size_bytes"]:
        raise ValueError("Encoder artifact size does not match its manifest")
    actual_sha256 = f"sha256:{file_sha256(artifact_path)}"
    if actual_sha256 != manifest["artifact_sha256"]:
        raise ValueError(
            "Encoder artifact file SHA-256 mismatch: "
            f"recorded={manifest['artifact_sha256']} actual={actual_sha256}"
        )
    return manifest


def load_encoder_ema_artifact(
    artifact_path: str | Path,
    *,
    expected_checkpoint_fingerprint: str,
    expected_cache_fingerprint: str,
    device: str | torch.device = "cpu",
) -> LoadedEncoderArtifact:
    """Fully validate and load an encoder-only EMA artifact.

    The caller must provide the deployment's expected 400k checkpoint and cache
    fingerprints. Merely trusting the values recorded inside the artifact would
    permit an otherwise valid encoder to be paired with the wrong VLA prefix.
    """

    path = Path(artifact_path).expanduser().resolve()
    manifest_path = artifact_manifest_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Encoder artifact does not exist: {path}")
    _require_sha256(expected_checkpoint_fingerprint, "expected_checkpoint_fingerprint")
    _require_sha256(expected_cache_fingerprint, "expected_cache_fingerprint")
    manifest = _read_and_validate_manifest(path, manifest_path)
    payload = _load_torch_payload(path)
    _require_exact_keys(payload, _ARTIFACT_ROOT_KEYS, "artifact")

    metadata = payload["metadata"]
    state = payload["encoder"]
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("Encoder artifact metadata and encoder state must be dictionaries")
    _require_exact_keys(metadata, _METADATA_KEYS, "artifact.metadata")
    if metadata["schema_version"] != ENCODER_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported encoder artifact schema_version")
    if metadata["artifact_kind"] != ENCODER_ARTIFACT_KIND:
        raise ValueError("Unexpected encoder artifact kind")
    if metadata["architecture"] != ENCODER_ARCHITECTURE:
        raise ValueError("Unexpected encoder artifact architecture")

    encoder_config_data = metadata["encoder_config"]
    if not isinstance(encoder_config_data, dict):
        raise ValueError("artifact.metadata.encoder_config must be a dictionary")
    _require_exact_keys(
        encoder_config_data,
        set(StrictEncoderConfig.__dataclass_fields__),
        "artifact.metadata.encoder_config",
    )
    encoder_config = StrictEncoderConfig(**encoder_config_data)
    encoder_config.validate()

    lineage = metadata["representation_lineage"]
    if not isinstance(lineage, dict):
        raise ValueError("artifact.metadata.representation_lineage must be a dictionary")
    _require_exact_keys(lineage, _LINEAGE_KEYS, "artifact.metadata.representation_lineage")
    if canonical_json_sha256(lineage) != canonical_json_sha256(manifest["representation_lineage"]):
        raise ValueError("Artifact and manifest representation lineage differ")
    if lineage["checkpoint_fingerprint"] != expected_checkpoint_fingerprint:
        raise ValueError(
            "Encoder artifact checkpoint fingerprint does not match deployment: "
            f"artifact={lineage['checkpoint_fingerprint']} "
            f"expected={expected_checkpoint_fingerprint}"
        )
    if lineage["cache_fingerprint"] != expected_cache_fingerprint:
        raise ValueError(
            "Encoder artifact cache fingerprint does not match deployment: "
            f"artifact={lineage['cache_fingerprint']} expected={expected_cache_fingerprint}"
        )
    if lineage["feature_tap"] != FEATURE_TAP or lineage["processor_mode"] != PROCESSOR_MODE:
        raise ValueError("Encoder artifact records a non-serving-equivalent prefix source")

    source_metadata = metadata["source_checkpoint"]
    if not isinstance(source_metadata, dict):
        raise ValueError("artifact.metadata.source_checkpoint must be a dictionary")
    _require_exact_keys(source_metadata, _SOURCE_METADATA_KEYS, "source_checkpoint")
    if source_metadata["schema_version"] != SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Encoder source checkpoint schema does not match")
    if source_metadata["architecture"] != SOURCE_ARCHITECTURE:
        raise ValueError("Encoder source checkpoint architecture does not match")
    _require_int(source_metadata["step"], "source_checkpoint.step", minimum=1)
    _require_int(source_metadata["size_bytes"], "source_checkpoint.size_bytes", minimum=1)
    _require_sha256(source_metadata["sha256"], "source_checkpoint.sha256")
    if source_metadata["sha256"] != manifest["source_checkpoint_sha256"]:
        raise ValueError("Artifact and manifest source checkpoint SHA-256 differ")
    if source_metadata["size_bytes"] != manifest["source_checkpoint_size_bytes"]:
        raise ValueError("Artifact and manifest source checkpoint sizes differ")
    if not isinstance(source_metadata["ema_decay"], (int, float)) or not (
        0 < float(source_metadata["ema_decay"]) < 1
    ):
        raise ValueError("source_checkpoint.ema_decay must be in (0, 1)")

    metadata_sha256 = canonical_json_sha256(metadata)
    if metadata_sha256 != manifest["metadata_sha256"]:
        raise ValueError("Encoder artifact metadata SHA-256 mismatch")
    expected_shapes = _encoder_state_shapes(encoder_config)
    _validate_tensor_state(state, expected_shapes, "artifact.encoder")
    state_sha256 = tensor_state_sha256(state)
    if state_sha256 != metadata["encoder_state_sha256"]:
        raise ValueError("Encoder tensor-state SHA-256 does not match artifact metadata")
    if state_sha256 != manifest["encoder_state_sha256"]:
        raise ValueError("Encoder tensor-state SHA-256 does not match artifact manifest")

    # Constructing on meta avoids allocating and then replacing another 1.5 GiB FP32
    # parameter set for the production configuration.
    with torch.device("meta"):
        encoder = StrictRLTokenEncoder(encoder_config)
    incompatible = encoder.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:  # pragma: no cover
        raise RuntimeError(f"Unexpected strict load result: {incompatible}")
    encoder.to(device=torch.device(device), dtype=torch.float32)
    encoder.eval()
    return LoadedEncoderArtifact(
        encoder=encoder,
        metadata=metadata,
        artifact_path=path,
        manifest_path=manifest_path,
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Export encoder-only EMA weights")
    export_parser.add_argument("source_checkpoint")
    export_parser.add_argument("output_path")
    export_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="Verify and load an encoder artifact")
    verify_parser.add_argument("artifact_path")
    verify_parser.add_argument("--expected-checkpoint-fingerprint", required=True)
    verify_parser.add_argument("--expected-cache-fingerprint", required=True)
    verify_parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    if args.command == "export":
        manifest = export_encoder_ema_artifact(
            args.source_checkpoint,
            args.output_path,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return
    loaded = load_encoder_ema_artifact(
        args.artifact_path,
        expected_checkpoint_fingerprint=args.expected_checkpoint_fingerprint,
        expected_cache_fingerprint=args.expected_cache_fingerprint,
        device=args.device,
    )
    print(json.dumps(loaded.metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
