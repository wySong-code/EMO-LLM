#!/usr/bin/env bash
# Launch on the same Linux server and conda environment used by the 7B SFT run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PARENT=$(cd "$ROOT/.." && pwd)
PY=${DAPO_PYTHON:-$HOME/miniconda3/envs/Emo-LLM/bin/python}
BASE=${DAPO_BASE:-$HOME/MM-LLMs-Emotion-main/models/Qwen2.5-Omni-7B}
SFT_EXPERIMENT=$PARENT/2026-10-30_23-00-Omni7B-SFT-GRPO
ADAPTER_DEFAULT=$SFT_EXPERIMENT/sft_run/adapter_best
if [[ ! -f "$ADAPTER_DEFAULT/adapter_config.json" ]]; then
    ADAPTER_DEFAULT=$SFT_EXPERIMENT/sft_run/train/checkpoints/checkpoint-1638
fi
ADAPTER=${DAPO_ADAPTER:-$ADAPTER_DEFAULT}
DATA=${DAPO_DATA:-$PARENT/training_dataset/Dataset}
CACHE=$ROOT/cache/media_cache
RUN_TAG=${DAPO_RUN_TAG:-dapo7b}
PROFILE=${DAPO_REWARD_PROFILE:-score_focused}
DECISION_BONUS=${DAPO_DECISION_BONUS:-0.5}
SMOKE=$ROOT/preflight/gpu_smoke_$RUN_TAG
OUT=$ROOT/runs/$RUN_TAG

export PYTHONPATH="$HOME/grpo_mm/envs/tf457:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled

for required in "$PY" "$BASE/config.json" "$ADAPTER/adapter_config.json" "$DATA/train.jsonl" "$DATA/val.jsonl"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 2
    fi
done
if [[ -e "$OUT" || -e "$SMOKE" ]]; then
    echo "Existing output or smoke directory; choose a fresh experiment path: $OUT / $SMOKE" >&2
    exit 2
fi

mkdir -p "$ROOT/logs" "$CACHE" "$ROOT/preflight" "$ROOT/runs"
SHARED_CACHE=$SFT_EXPERIMENT/grpo_run/media_cache
if [[ -d "$SHARED_CACHE" ]]; then
    cp -al "$SHARED_CACHE"/. "$CACHE"/ 2>/dev/null || true
fi

"$PY" "$ROOT/code/test_dapo.py"
"$PY" "$ROOT/code/train_dapo.py" --dataset-root "$DATA" --check-data
CARDS=$("$PY" "$ROOT/scripts/pick_cards.py" 2 800)
export CUDA_VISIBLE_DEVICES="$CARDS"
echo "Selected physical GPUs $CARDS; policy=cuda:0 reference=cuda:1"
"$PY" "$ROOT/code/train_dapo.py" --dataset-root "$DATA" --check-env

"$PY" -u "$ROOT/code/train_dapo.py" \
    --base "$BASE" --adapter "$ADAPTER" --dataset-root "$DATA" \
    --code "$ROOT/code/sft" --media-cache "$CACHE" \
    --policy-device 0 --reference-device 1 --out "$SMOKE" \
    --reward-profile "$PROFILE" --decision-advantage-bonus "$DECISION_BONUS" \
    --smoke-test > "$ROOT/logs/gpu_smoke_$RUN_TAG.log" 2>&1 || {
        tail -n 80 "$ROOT/logs/gpu_smoke_$RUN_TAG.log" >&2
        exit 4
    }

"$PY" -u "$ROOT/code/train_dapo.py" \
    --base "$BASE" --adapter "$ADAPTER" --dataset-root "$DATA" \
    --code "$ROOT/code/sft" --media-cache "$CACHE" \
    --policy-device 0 --reference-device 1 --out "$OUT" \
    --steps 300 --candidates 8 --max-conflict-attempts 4 --max-aligned-attempts 2 \
    --screen-train --screen-batch-size 4 --hard-prob 0.75 \
    --original-conflict-prob 0.8 --conflict-advantage-scale 1.5 \
    --reward-profile "$PROFILE" --decision-advantage-bonus "$DECISION_BONUS" \
    --ppo-epochs 2 --lr 2e-6 --kl-beta 0.02 \
    --clip-low 0.2 --clip-high 0.28 --anchor-weight 0.05 \
    --temperature 1.1 --top-p 0.95 --max-new-tokens 128 \
    --eval-every 50 --eval-batch-size 4 --f1-drop-guard 0.01 --seed 42 \
    > "$ROOT/logs/dapo_train_$RUN_TAG.log" 2>&1 || {
        tail -n 80 "$ROOT/logs/dapo_train_$RUN_TAG.log" >&2
        exit 5
    }

cat "$OUT/best.json"
