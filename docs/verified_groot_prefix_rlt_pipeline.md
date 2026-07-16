# Verified GR00T-prefix RLT pipeline

This is the canonical pipeline for training the Groot-RLT encoder and decoder.
It was validated end to end on `node2` on 2026-07-15 using a fine-tuned GR00T
checkpoint at step 400,000. Future representation runs should follow the same
sequence and must pass the same gates before producing replay or actor/critic
artifacts.

## Non-negotiable contracts

1. The prefix source is the target fine-tuned GR00T policy checkpoint, not the
   standalone Cosmos/VLM weights and not an unrelated base checkpoint.
2. `--base-model-path` and `--processor-path` point to the same saved GR00T
   checkpoint. The cache records hashes of the model and processor files.
3. Prefix features are tapped at `raw_backbone_pre_action_head` with the model
   and processor in evaluation mode. The action head must not mutate the
   features before the RLT encoder consumes them.
4. The token contract is `image / uniform / 192`, stored as bfloat16 with no
   secondary token selection after cache construction.
5. The representation architecture is
   `openpi_rlt_strict_cross_attention_v1`:

   - the encoder uses one learned query to cross-attend to the GR00T prefix;
   - the decoder uses learned positional queries and cross-attends only to the
     single RL token;
   - the decoder never receives ground-truth prefix embeddings.

6. A prefix cache belongs to exactly one dataset, GR00T checkpoint, processor,
   instruction contract, token contract, and feature tap. Never reuse or append
   to a cache after any of those inputs changes.
7. The 400k GR00T policy remains frozen. Training updates only the RLT encoder
   and decoder.
8. Downstream execution uses row-first rot6d:
   `[r00, r01, r02, r10, r11, r12]`.

The checkpoint and cache validators fail closed when these contracts do not
match. Do not bypass them by editing manifests or substituting paths with
similar filenames.

## Environment and run identity

Run these commands in the Python 3.10 Isaac-GR00T environment with `groot_rlt`
installed. Use a new immutable output name for every experiment.

```bash
export RLT_ROOT=/workspace/RLT
export GROOT_ROOT=/workspace/Isaac-GR00T

export TRAIN_DATASET=<prepared-training-dataset>
export HOLDOUT_DATASET=<different-date-holdout-dataset>
export MODALITY_CONFIG=<matching-modality-config.py>
export GROOT_CKPT=<fine-tuned-groot-checkpoint>
export VLM_CKPT=<Cosmos-Reason2-2B>
export INSTRUCTION=<canonical-task-instruction>

export RUN_NAME=<unique-run-name>
export PREFIX_CACHE="$RLT_ROOT/outputs/cache/vl_embeddings/$RUN_NAME"
export RUN_DIR="$RLT_ROOT/outputs/runs/$RUN_NAME/encoder_decoder"
export EVAL_DIR="$RLT_ROOT/outputs/evaluations/$RUN_NAME"
export HOLDOUT_PREFIX="$EVAL_DIR/holdout_true_prefixes"
```

If the dataset already contains the canonical task text, omit `--instruction`
consistently from every stage. If an override is required, use the exact same
string everywhere; it becomes part of the cache lineage.

`PREFIX_CACHE`, `RUN_DIR`, `EVAL_DIR`, and downstream bridge paths should not
exist before a new run. This prevents historical artifacts from being mixed
into the new lineage.

## Stage 1: materialize the true GR00T prefix

```bash
groot-rlt train-token \
  --groot-repo-path "$GROOT_ROOT" \
  --precompute-vl-embeddings \
  --dataset-dir "$TRAIN_DATASET" \
  --modality-config-path "$MODALITY_CONFIG" \
  --base-model-path "$GROOT_CKPT" \
  --processor-path "$GROOT_CKPT" \
  --vlm-model-path "$VLM_CKPT" \
  --instruction "$INSTRUCTION" \
  --embedding-cache-dir "$PREFIX_CACHE" \
  --token-scope image \
  --token-sampling uniform \
  --max-vl-tokens 192 \
  --cache-dtype bfloat16 \
  --episode-sampling-rate 1.0 \
  --shard-size 1024 \
  --video-backend torchcodec \
  --dataloader-num-workers 0 \
  --device cuda \
  --local-files-only
```

Do not pass `--overwrite-cache` during a normal run. After completion, retain
`manifest.json` and all cache shards. The manifest must report:

```text
representation_source = groot_checkpoint_backbone
feature_tap           = raw_backbone_pre_action_head
processor_mode        = eval
token_scope           = image
token_sampling        = uniform
max_vl_tokens         = 192
cache_dtype           = bfloat16
```

## Stage 2: train the strict encoder and decoder

The validated baseline uses FP32 RLT training and EMA 0.99. The cached prefix
remains bfloat16.

```bash
groot-rlt train-token \
  --groot-repo-path "$GROOT_ROOT" \
  --dataset-dir "$TRAIN_DATASET" \
  --modality-config-path "$MODALITY_CONFIG" \
  --base-model-path "$GROOT_CKPT" \
  --processor-path "$GROOT_CKPT" \
  --vlm-model-path "$VLM_CKPT" \
  --instruction "$INSTRUCTION" \
  --embedding-cache-dir "$PREFIX_CACHE" \
  --output-dir "$RUN_DIR" \
  --max-steps 10000 \
  --batch-size 32 \
  --learning-rate 2.5e-5 \
  --min-learning-rate 2.5e-6 \
  --weight-decay 1e-10 \
  --warmup-steps 1000 \
  --lr-decay-steps 30000 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --ema-decay 0.99 \
  --grad-clip 1.0 \
  --fail-on-nonfinite \
  --save-steps 5000 \
  --token-scope image \
  --token-sampling uniform \
  --max-vl-tokens 192 \
  --model-dim 2048 \
  --rl-token-dim 2048 \
  --encoder-layers 2 \
  --decoder-layers 2 \
  --num-heads 8 \
  --mlp-ratio 4.0 \
  --dropout 0.0 \
  --decoder-cross-attention \
  --no-decoder-prefix-corruption \
  --no-autoencoder-bf16 \
  --device cuda \
  --seed 42 \
  --local-files-only \
  --use-swanlab \
  --swanlab-project groot-rlt \
  --swanlab-experiment-name "$RUN_NAME" \
  --swanlab-tags strict-openpi-rlt,true-groot-prefix,image192,bf16-cache,fp32-ae,ema099,10k
```

The expected checkpoints are `005000.pt` and `010000.pt`. A valid final
checkpoint has schema version 2, architecture
`openpi_rlt_strict_cross_attention_v1`, the GR00T checkpoint fingerprint, and
the prefix-cache fingerprint. Preserve `training_config.json` and the complete
local training log even when SwanLab is enabled.

## Stage 3: mandatory checkpoint audit

```bash
python -m groot_rlt.representation.audit_rl_token_checkpoint \
  --checkpoint "$RUN_DIR/010000.pt" \
  --cache-manifest "$PREFIX_CACHE/manifest.json" \
  --model-path "$GROOT_CKPT" \
  --expected-step 10000 \
  --expected-token-scope image \
  --expected-token-sampling uniform \
  --expected-max-vl-tokens 192 \
  --expected-model-dim 2048 \
  --expected-cache-dtype bfloat16 \
  --verify-optimizer-finite \
  --verify-cache-sha256 \
  --output-json "$EVAL_DIR/checkpoint_audit.json"
```

Required result: `verdict = pass`. This scans raw, EMA, and optimizer tensors
for non-finite values and re-hashes every cache shard.

## Stage 4: mandatory golden parity

```bash
groot-rlt verify-prefix-parity \
  --groot-repo-path "$GROOT_ROOT" \
  --model-path "$GROOT_CKPT" \
  --processor-path "$GROOT_CKPT" \
  --vlm-model-path "$VLM_CKPT" \
  --dataset-dir "$TRAIN_DATASET" \
  --modality-config-path "$MODALITY_CONFIG" \
  --instruction "$INSTRUCTION" \
  --token-scope image \
  --token-sampling uniform \
  --max-vl-tokens 192 \
  --cache-dtype bfloat16 \
  --rl-token-checkpoint "$RUN_DIR/010000.pt" \
  --denoise-steps 32 \
  --verify-reference \
  --device cuda \
  --output-dir "$EVAL_DIR/golden_parity"
```

Required result: `verdict = pass`; processor inputs, raw backbone prefix,
packed `[1, 192, 2048]` prefix, masks, and cache round trip must be exact with
`max_abs = 0`. The GR00T reference must also preserve the expected full and
projected action shapes.

## Stage 5: mandatory ablation and dataset UMAP

```bash
groot-rlt evaluate-token \
  --checkpoint "$RUN_DIR/010000.pt" \
  --embedding-cache-dir "$PREFIX_CACHE" \
  --device cuda \
  --no-autoencoder-bf16 \
  --fail-on-not-ready \
  --output-json "$EVAL_DIR/ablation.json"

groot-rlt visualize-token \
  --groot-repo-path "$GROOT_ROOT" \
  --source dataset \
  --checkpoint "$RUN_DIR/010000.pt" \
  --dataset-dir "$HOLDOUT_DATASET" \
  --modality-config-path "$MODALITY_CONFIG" \
  --base-model-path "$GROOT_CKPT" \
  --processor-path "$GROOT_CKPT" \
  --vlm-model-path "$VLM_CKPT" \
  --instruction "$INSTRUCTION" \
  --episode-sampling-rate 1.0 \
  --token-scope image \
  --token-sampling uniform \
  --max-vl-tokens 192 \
  --reducer umap \
  --no-allow-pca-fallback \
  --device cuda \
  --output-dir "$EVAL_DIR/dataset_umap"
```

The ablation must report `ready = true`: zeroing or shuffling the RL token must
damage reconstruction, and zeroing image tokens must change the encoder output.
UMAP must use dataset inputs, not cached training rows, and must not silently
fall back to PCA. Treat UMAP as a visualization; quantitative readiness comes
from the ablation and holdout gates.

## Stage 6: mandatory true-prefix holdout

The holdout must come from different raw episodes, preferably a different
collection date. Extraction rejects any overlap in `raw_episode_id`.

```bash
python -m groot_rlt.representation.extract_true_prefix_holdout \
  --dataset-dir "$HOLDOUT_DATASET" \
  --training-dataset-dir "$TRAIN_DATASET" \
  --output-dir "$HOLDOUT_PREFIX" \
  --modality-config-path "$MODALITY_CONFIG" \
  --base-model-path "$GROOT_CKPT" \
  --processor-path "$GROOT_CKPT" \
  --vlm-model-path "$VLM_CKPT" \
  --instruction "$INSTRUCTION" \
  --video-backend torchcodec \
  --frame-stride 5 \
  --cache-dtype bfloat16 \
  --device cuda

python -m groot_rlt.representation.evaluate_true_prefix_holdout \
  --legacy-evaluator "$HOLDOUT_EVALUATOR" \
  --fail-on-not-ready \
  --checkpoint-5k "$RUN_DIR/005000.pt" \
  --checkpoint-10k "$RUN_DIR/010000.pt" \
  --training-config "$RUN_DIR/training_config.json" \
  --training-log "$TRAIN_LOG" \
  --training-cache-dir "$PREFIX_CACHE" \
  --holdout-prefix-dir "$HOLDOUT_PREFIX" \
  --output-dir "$EVAL_DIR/holdout_evaluation" \
  --train-samples 4096 \
  --batch-size 16 \
  --seed 42 \
  --device cuda
```

`HOLDOUT_EVALUATOR` is an immutable evaluator artifact. Record its SHA-256 in
the result; the 2026-07-15 baseline used
`65de43fcf1d07a8faad538721252310ec7fe45c041f19592939c672cff951453`.

Required result: `verdict = trained_and_functional` and every gate is true.
The gate set includes finite/non-collapsed latents, positive centered holdout
R2, 10k not worse than 5k, perturbation confidence intervals, decoder target
invariance, and UMAP trustworthiness for two seeds.

## Stage 7: export the serving encoder

```bash
python -m groot_rlt.representation.encoder_artifact export \
  "$RUN_DIR/010000.pt" \
  "$RLT_ROOT/outputs/artifacts/$RUN_NAME/encoder_ema.pt"

python -m groot_rlt.representation.encoder_artifact verify \
  "$RLT_ROOT/outputs/artifacts/$RUN_NAME/encoder_ema.pt" \
  --expected-checkpoint-fingerprint <from-cache-manifest> \
  --expected-cache-fingerprint <from-cache-manifest> \
  --device cpu
```

Use EMA encoder weights for serving. Keep the sidecar manifest with the
artifact. The full `010000.pt` remains the authoritative encoder/decoder
training checkpoint.

## Downstream admission rule

Only after stages 3 through 7 pass may the encoder be used for LeRobot v3
feature generation, replay bridge construction, or actor/critic training. The
v3 bridge must continue to enforce:

- full 26D state/reference for frozen GR00T inference and provenance;
- projected 19D `eef9 + hand10` proprio and executed action for the learner;
- no arm7 values in actor/critic inputs or outputs;
- dataset, checkpoint, processor, encoder, prefix-cache, VLM, token, and camera
  fingerprints pinned before feature generation.

The concrete v3 interface and action contracts are documented in
[`groot_teleop_integration.md`](groot_teleop_integration.md) and the package
READMEs.

## Verified 2026-07-15 baseline

The successful `node2` run established the following reference results:

| Check | Result |
| --- | --- |
| Training | 10,000 steps; final logged loss 1.614316 |
| Final checkpoint SHA-256 | `9211d4f6d76decb9460134d5f154044fd4fdff95526add757cc223b8ee52e7fc` |
| Golden parity | pass; all exact comparisons `max_abs = 0` |
| Ablation | normal MSE 1.4643; zero ratio 2.5925; shuffle ratio 3.2208 |
| Holdout | 1,586 samples from 38 unseen episodes; no raw episode overlap |
| Holdout quality | 10k EMA MSE 2.5673; centered R2 0.1024; Recall@1 0.9830 |
| Holdout verdict | `trained_and_functional`; 7/7 gates passed |
| UMAP trustworthiness | 0.9805 and 0.9794 for the two seeds |
| v3 replay | 759 source frames; 757 steps; 610 policy and 149 correction frames |
| Actor/critic smoke | 361 transitions; 20 finite steps; non-production smoke passed |

The v3 sample contained two successful episodes and no failed episodes. It
validated the wiring and contracts but is not sufficient evidence that the
critic is production-trained. Production actor/critic training requires a
larger outcome-diverse dataset with failures, recoveries, and interventions.

## Retention checklist

Retain these together for every accepted representation run:

- prefix-cache manifest and shard hashes;
- `005000.pt`, `010000.pt`, `training_config.json`, and the local log;
- checkpoint audit, golden parity, ablation, UMAP, and holdout summaries;
- holdout manifest proving zero raw-episode overlap;
- encoder EMA artifact and its sidecar manifest;
- all dataset/model/processor/token/camera fingerprints used downstream;
- SwanLab run URL when enabled, without treating cloud upload success as a
  substitute for the local log and checkpoint hashes.

