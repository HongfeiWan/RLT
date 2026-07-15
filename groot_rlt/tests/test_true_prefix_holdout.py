from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from groot_rlt.integration.artifact_lineage import file_sha256
from groot_rlt.representation.evaluate_true_prefix_holdout import _install_strict_loaders


@dataclass
class _EvalSet:
    name: str
    prefixes: torch.Tensor
    metadata: pd.DataFrame
    state: np.ndarray | None = None
    action: np.ndarray | None = None


def _write_prefix_shard(path: Path, samples: int) -> dict[str, object]:
    packed = torch.arange(samples * 192 * 2048, dtype=torch.float32).reshape(
        samples, 192, 2048
    )
    torch.save(
        {
            "packed": packed.to(torch.bfloat16),
            "packed_mask": torch.ones(samples, 192, dtype=torch.bool),
            "packed_image_mask": torch.ones(samples, 192, dtype=torch.bool),
            "token_counts": torch.full((samples,), 192, dtype=torch.int32),
            "selected_counts": torch.full((samples,), 192, dtype=torch.int32),
        },
        path,
    )
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "num_samples": samples,
        "num_valid_tokens": samples * 192,
    }


def test_strict_eval_loaders_preserve_direct_bf16_prefixes(tmp_path: Path) -> None:
    training_dir = tmp_path / "train"
    holdout_dir = tmp_path / "holdout"
    training_dir.mkdir()
    holdout_dir.mkdir()
    training_shard = _write_prefix_shard(training_dir / "shard_000000.pt", 3)
    holdout_shard = _write_prefix_shard(holdout_dir / "prefix_shard_0000.pt", 2)

    metadata = pd.DataFrame(
        {
            "episode_index": [1, 2],
            "frame_index": [0, 0],
            "progress": [0.0, 0.0],
        }
    )
    metadata_path = holdout_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    robot_path = holdout_dir / "robot_metadata.npz"
    np.savez_compressed(
        robot_path,
        state=np.zeros((2, 26), dtype=np.float32),
        action=np.zeros((2, 19), dtype=np.float32),
    )
    training_manifest = {
        "num_samples": 3,
        "shards": [training_shard],
    }
    holdout_manifest = {
        "num_samples": 2,
        "shards": [holdout_shard],
        "metadata_csv_sha256": file_sha256(metadata_path),
        "robot_metadata_npz_sha256": file_sha256(robot_path),
    }
    module = SimpleNamespace(EvalSet=_EvalSet)
    _install_strict_loaders(
        module,
        training_manifest=training_manifest,
        holdout_manifest=holdout_manifest,
        expected_config={},
        checkpoint_fingerprint="sha256:model",
        cache_fingerprint="sha256:cache",
    )

    holdout = module.load_holdout(holdout_dir)
    training = module.load_training_sample(training_dir, 2, 42)

    assert holdout.prefixes.dtype == torch.bfloat16
    assert training.prefixes.dtype == torch.bfloat16
    assert holdout.prefixes.shape == (2, 192, 2048)
    assert training.prefixes.shape == (2, 192, 2048)
    assert holdout.state.shape == (2, 26)
    assert holdout.action.shape == (2, 19)
