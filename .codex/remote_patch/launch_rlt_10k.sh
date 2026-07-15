#!/usr/bin/env bash
set -euo pipefail

name=rlt_nero_m2_ckpt400k_image192_fp32_ema099_10k_20260714
image=sha256:b1f36b19921c0f1713d31a4b1a213da6d5a4b0edff4334c161378ebce065fb12

if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container already exists: $name" >&2
    exit 2
fi

docker run -d \
    --name "$name" \
    --pull=never \
    --gpus all \
    --ipc=host \
    --security-opt label=disable \
    --restart=no \
    -e PYTHONPATH=/workspace/RLT/groot_rlt/src:/workspace/Isaac-GR00T \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e GROOT_RLT_PROJECT_ROOT=/workspace/RLT \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e HOME=/container_home \
    -v /home/user/project/Isaac-GR00T:/workspace/Isaac-GR00T:ro \
    -v /home/user/project/Isaac-GR00T/docker_home:/container_home \
    -v /home/user/project/RLT:/workspace/RLT \
    -w /workspace/RLT \
    "$image" \
    bash -lc '
set -euo pipefail
PY=/opt/gr00t/.venv/bin/python
CACHE=/workspace/RLT/outputs/cache/vl_embeddings/nero_mission2_smooth_ckpt400k_tokens256
OUT=/workspace/RLT/outputs/runs/nero_mission2_smooth_ckpt400k_openpi_image192_fp32_ema099_10k_20260714/encoder_decoder
LOG=/workspace/RLT/outputs/logs/rlt_nero_mission2_smooth_ckpt400k_openpi_image192_fp32_ema099_10k_20260714.log
test -f "$CACHE/manifest.json"
mkdir -p "$OUT" "$(dirname "$LOG")"
if compgen -G "$OUT/*.pt" >/dev/null; then
    echo "Refusing non-fresh output: $OUT" >&2
    exit 2
fi
{
    echo "[$(date -Is)] architecture=openpi_rlt token_scope=image precision=fp32 ema=0.99 max_steps=10000 status=starting"
    "$PY" -m groot_rlt.cli train-token \
        --groot-repo-path /workspace/Isaac-GR00T \
        --dataset-dir /workspace/Isaac-GR00T/outputs/IsaacLab/nero/mission2/smooth \
        --modality-config-path /workspace/Isaac-GR00T/examples/IsaacLab/nero_right_l10_multiview_modality_config.py \
        --base-model-path /workspace/Isaac-GR00T/checkpoints/nero_right_l10_mission2_smooth_action_state_aug_vargrid_symzero_all_247_20260710_2125_400k/mission2-smooth-action-state-aug-vargrid-symzero-all-247-20260710_2125/checkpoint-400000 \
        --vlm-model-path /workspace/Isaac-GR00T/checkpoints/nvidia/Cosmos-Reason2-2B \
        --instruction "pick up the bottle with green cap and place it in the white rectangle area" \
        --embedding-cache-dir "$CACHE" \
        --output-dir "$OUT" \
        --max-steps 10000 \
        --batch-size 32 \
        --dataloader-num-workers 0 \
        --learning-rate 2.5e-5 \
        --min-learning-rate 2.5e-6 \
        --warmup-steps 1000 \
        --lr-decay-steps 30000 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --weight-decay 1e-10 \
        --grad-clip 1.0 \
        --ema-decay 0.99 \
        --fail-on-nonfinite \
        --log-steps 10 \
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
        --device cuda \
        --no-autoencoder-bf16 \
        --local-files-only \
        --use-swanlab \
        --swanlab-project groot-rlt \
        --swanlab-experiment-name nero_mission2_smooth_ckpt400k_openpi_image192_fp32_ema099_10k_20260714 \
        --swanlab-mode cloud \
        --swanlab-logdir /workspace/RLT/outputs/swanlab \
        --swanlab-tags openpi-rlt-repro,image-only,fp32,ema099,10k,nero,mission2,ckpt400k \
        --swanlab-log-steps 10 \
        --swanlab-log-model-steps 1000 \
        --no-swanlab-eval-ablation-on-checkpoint
    echo "[$(date -Is)] architecture=openpi_rlt status=complete"
} 2>&1 | tee -a "$LOG"
'
