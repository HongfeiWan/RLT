# Groot-RLT

`groot_rlt` is the primary Python 3.10 integration layer between Isaac-GR00T
N1.7 and this repository's RLT implementation. It keeps GR00T/Cosmos
dependencies out of the retained openpi Python 3.11 environment while exposing
the Machine-A contract used by `rlt_online_rl`.

The package contains three deliberately separate pieces:

1. GR00T/Cosmos RL-token representation training and diagnostics.
2. A strict NumPy/PyTorch episode, replay, and actor-critic implementation.
3. A GR00T feature server that returns `z_rl`, physical-space VLA reference
   chunks, and proprio to the existing online-RL runtime.

The first two pieces are usable without a robot. The third is a tested software
contract, not a claim that the current 26D robot/teleop path has passed an
end-to-end hardware rollout. See the root
[README](../README.md) and [Teleop integration contract](../docs/groot_teleop_integration.md)
for the current deployment boundary.

## Environment boundary

Do not install this package into the openpi root environment. openpi requires
Python 3.11 while Isaac-GR00T uses Python 3.10 and a different Transformers
version. Install it into the Isaac-GR00T environment:

```bash
cd /home/whf/Project/Isaac-GR00T
uv pip install --python .venv/bin/python \
  -e '/home/whf/Project/RLT/groot_rlt[groot,data,serve,dev]'

.venv/bin/groot-rlt --help
```

The GR00T checkout is resolved in this order:

1. `--groot-repo-path`
2. `GROOT_REPO_PATH`
3. a sibling `Isaac-GR00T` checkout
4. an installed `gr00t` package backed by a source checkout

An explicitly supplied invalid checkout fails immediately instead of silently
falling back to another repository.

## Package boundary

The migrated PyTorch implementation uses the `groot_rlt` namespace:

```python
from groot_rlt import RLTActor, RLTReplayBuffer, RLTTrainer, RLTTransition
```

It does not replace the JAX production runtime in `rlt_online_rl`:

- `groot_rlt.RLTTransition` is the strict, provenance-rich offline schema.
- `rlt_online_rl.replay.RLTTransition` is the compact service/runtime schema.
- Their checkpoints and transition objects are not interchangeable.

The PyTorch core has no dependency on GR00T model code. Importing
`groot_rlt` only requires NumPy and PyTorch. GR00T is loaded lazily by the
representation and serving entry points.

## Unified command

`groot-rlt` is the primary entry point. It lazily dispatches to the existing
tools, so displaying help does not load GR00T/Cosmos or require a GPU:

```bash
groot-rlt train-token --help
groot-rlt evaluate-token --help
groot-rlt precompute --help
groot-rlt visualize-token --help
groot-rlt serve-features --help
groot-rlt export-online-stats --help
```

The former `groot-rlt-train-token`, `groot-rlt-evaluate-token`,
`groot-rlt-precompute`, `groot-rlt-visualize-token`,
`groot-rlt-serve-features`, and `groot-rlt-export-online-stats` executables are
retained as compatibility aliases.

## Representation commands

These commands replace the former scripts under
`Isaac-GR00T/examples/IsaacLab`.

Example two-stage representation training:

```bash
groot-rlt train-token \
  --groot-repo-path /home/whf/Project/Isaac-GR00T \
  --precompute-vl-embeddings \
  --embedding-cache-dir /home/whf/Project/Isaac-GR00T/outputs/IsaacLab/vl_embedding_cache \
  --overwrite-cache \
  --dataset-dir <smooth-dataset> \
  --modality-config-path <modality-config.py> \
  --base-model-path <GR00T-N1.7-3B> \
  --vlm-model-path <Cosmos-Reason2-2B> \
  --episode-sampling-rate 1.0 \
  --device cuda \
  --load-bf16

groot-rlt train-token \
  --groot-repo-path /home/whf/Project/Isaac-GR00T \
  --embedding-cache-dir /home/whf/Project/Isaac-GR00T/outputs/IsaacLab/vl_embedding_cache \
  --output-dir /home/whf/Project/Isaac-GR00T/outputs/IsaacLab/vl_embedding_autoencoder \
  --max-steps 20000 \
  --batch-size 32 \
  --autoencoder-bf16 \
  --device cuda
```

The migrated precompute command no longer imports private helpers from
`examples.IsaacLab`; its checkpoint, LeRobot, image, and RTC helpers live under
`groot_rlt.integration`. `--embodiment-tag` is propagated to the checkpoint
processor instead of being hard-coded to `NEW_EMBODIMENT`.

## GR00T Machine-A feature server

The feature server uses one GR00T backbone pass for both the action chunk and
the RL token:

```bash
groot-rlt serve-features \
  --groot-repo-path /home/whf/Project/Isaac-GR00T \
  --model-path <checkpoint-directory> \
  --processor-path <processor-directory> \
  --vlm-model-path <Cosmos-Reason2-2B> \
  --rl-token-checkpoint <rl-token-checkpoint.pt> \
  --embodiment-tag NEW_EMBODIMENT \
  --z-dim 2048 \
  --chunk-len 10 \
  --action-dim 26 \
  --proprio-dim 26 \
  --denoise-steps 32 \
  --host 0.0.0.0 \
  --port 8000
```

The WebSocket protocol is compatible with
`rlt_online_rl.inference.MachineAFeatureClient`:

```text
request observation
  -> z_rl      float32 [2048]
  -> ref_chunk float32 [chunk_len, 26]
  -> proprio   float32 [26]
```

`/healthz` is provided for readiness checks. The current GR00T backend reports
`supports_batch=false`: a batch wrapper would only run expensive 32-step GR00T
inferences serially, so the online client deliberately falls back to individual
requests instead of risking its default receive timeout.

The server accepts either native GR00T observations or the RLT flat wire shape
`{"images": ..., "state": ..., "prompt": ...}`. For Nero, a flat 26D state is
split strictly as `eef_9d[9] + hand_joint_pos[10] + arm_joint_pos[7]`. Use
`--image-key SOURCE=TARGET` for camera-name differences and
`--flat-state-field KEY=DIM` for any non-Nero layout.

This implementation intentionally calls the N1.7
`prepare_input -> backbone -> action_head` path so `z_rl` is encoded from the
raw backbone features before the action head mutates them. These are
version-sensitive GR00T APIs; a future GR00T release or checkpoint must pass a
real-checkpoint smoke test before the server is used for collection.

`ref_chunk` is processor-decoded physical-space action. Do not treat it as an
already normalized actor input. `rlt_online_rl.ActionRepresentationAdapter`
must apply the task's action stats exactly once.

`--denoise-steps 32` reproduces the currently validated Nero flow-sampling
setting; omitting it preserves the checkpoint value. The feature provider is
intentionally stateless and does not apply RTC. RTC/history stitching belongs
on the execution side: making feature extraction stateful would corrupt replay
feature reconstruction when observations are requested out of order.

## Nero 26D configuration

The online runtime now accepts a server-provided `proprio`, which allows nested
GR00T observations instead of requiring the old flat `observation["state"]`.
It also supports configurable, non-contiguous delta channels:

```yaml
experiment:
  rl:
    action_dim: 26
    proprio_dim: 26
    z_dim: 2048
    chunk_len: 10
    action_representation: abs
    action_norm_stats_path: /absolute/path/to/action_stats.json
    action_layout_hash: sha256:<copy from exported stats>
    proprio_layout_hash: sha256:<copy from Machine-A metadata>
    delta_action_indices: []
```

For a delta representation, list only channels whose actions and proprio share
the same physical coordinates, for example:

```yaml
delta_action_indices: [0, 1, 2, 19, 20, 21, 22, 23, 24, 25]
```

The provided Nero exporter intentionally emits `abs` stats. A `delta_chunk`
experiment must compute q01/q99 from the transformed chunk labels themselves;
renaming absolute stats as delta stats is invalid and is rejected by the loader.

The historical Agilex behavior is unchanged when the field is omitted: the
first `min(6, action_dim)` channels are used.

State-as-action is supported. The essential invariant is that `proprio`, VLA
reference, replay action, normalization stats, actor output, and deployment
denormalization all use the same 26D ordering.

Export versioned Nero online-action stats directly from a prepared LeRobot
dataset. The exporter validates the state-as-action aliases and writes the
strict `eef_9d[9] + hand_joint_target[10] + arm_joint_target[7]` layout:

```bash
groot-rlt export-online-stats \
  --dataset-dir /path/to/mission2/smooth \
  --normalization-mode symmetric_quantile \
  --output /path/to/nero_rlt_online_action_stats.json
```

`quantile` uses `q01`/`q99` as bounds. `symmetric_quantile` uses
`scale[d] = max(abs(q01[d]), abs(q99[d]), eps)` and writes the effective bounds
to `min`/`max`; the original quantiles and observed extrema remain in the file
for auditing. Each export also records canonical channel names, a layout hash,
and source fingerprints.

The online loader follows `normalization.lower_key`/`upper_key`, so symmetric
exports actually use `min`/`max` rather than silently falling back to the old
asymmetric `q01`/`q99` bounds. It accepts both per-dimension `[D]` and
per-horizon `[H,D]` bounds and verifies representation, horizon, dimension, and
layout hash before training or inference.

## Replay and training flow

```text
episode sidecars
  -> episode_schema.py
  -> episode_transition_builder.py
  -> replay_buffer.py / collate.py
  -> networks.py + train.py
  -> trainer.py
```

The strict schema preserves decision/sample time, VLA horizon, valid masks,
critical phase, behavior/reference provenance, intervention masks, and sparse
terminal labels. See [episode_schema.md](episode_schema.md) for the on-disk
format.

Teleop's current RLT sidecar exporter is only a handoff format: its RL-token
tensors are empty and its VLA reference keys are unset. Enrich and validate
those sidecars with the exact frozen GR00T/RL-token checkpoints before building
warmup replay.

## Historical artifact compatibility

Warmup replay and trainer checkpoints written before migration may contain
pickle paths such as `gr00t.data.rlt.replay_schema.RLTTransition`. The new
loaders temporarily map those paths to `groot_rlt.*`; no compatibility shim is
left in Isaac-GR00T.

After loading an old artifact, save it again to rewrite it with the new module
paths.

## Tests

Run the migrated core and integration-contract tests in the GR00T environment:

```bash
cd /home/whf/Project/RLT/groot_rlt
PYTHONPATH=src /home/whf/Project/Isaac-GR00T/.venv/bin/python \
  -m pytest tests -q -p no:cacheprovider
```

Run the JAX online runtime tests in its own Python 3.10 environment:

```bash
cd /home/whf/Project/RLT/rlt_online_rl
python -m pytest tests -q -p no:cacheprovider
```
