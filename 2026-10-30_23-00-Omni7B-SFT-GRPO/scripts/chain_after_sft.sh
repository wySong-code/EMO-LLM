#!/bin/bash
# Server-side chain: wait for SFT, emit CSV #1, then run the step50 GRPO and emit CSV #2.
#
# Why a chain: the SFT run alone is long enough that the next stage cannot be started by
# hand afterwards. Everything after the SFT is therefore queued as one detached sequence
# instead of being launched step by step.
#
# Order matters. CSV #1 is generated first (about ten minutes) so the two later stages
# never race for the same device: pick_cards.py would otherwise hand the reference model and
# the submission job the same device during their load windows.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/miniconda3/envs/Emo-LLM/bin/python
LOG=$ROOT/logs/chain.log
SFT_STATUS=$ROOT/sft_run/train/status.json

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "chain started, waiting for $SFT_STATUS"

# --- 1. wait for SFT ---------------------------------------------------------------
while true; do
  if [ -f "$SFT_STATUS" ] && grep -q '"stage": *"complete"' "$SFT_STATUS"; then
    say "SFT complete"
    break
  fi
  if ! pgrep -f 'train_omni_combined_sft.py' > /dev/null; then
    say "FATAL: SFT process gone and status is not complete"
    exit 1
  fi
  sleep 120
done

ADAPTER_SFT=$("$PY" -c "import json;print(json.load(open('$ROOT/sft_run/train/best_checkpoint.json'))['path'])")
say "sft best checkpoint = $ADAPTER_SFT"

# --- 2. CSV #1: pure SFT -----------------------------------------------------------
if [ ! -f "$ROOT/submissions/sft7b/submission.csv" ]; then
  say "start CSV #1 (sft7b)"
  bash "$ROOT/scripts/run_submit.sh" "$ADAPTER_SFT" sft7b >> "$LOG" 2>&1
  say "CSV #1 rc=$?"
else
  say "CSV #1 already present"
fi

# --- 3. GRPO, the step50 recipe ----------------------------------------------------
if [ ! -f "$ROOT/grpo_run/pilot_01/checkpoints/step_0050/adapter_model.safetensors" ]; then
  say "start GRPO"
  bash "$ROOT/scripts/run_grpo.sh" "$ADAPTER_SFT" >> "$LOG" 2>&1
  say "GRPO rc=$?"
else
  say "GRPO step_0050 already present"
fi

# --- 4. CSV #2: the step50 GRPO checkpoint -----------------------------------------
ADAPTER_GRPO=$ROOT/grpo_run/pilot_01/checkpoints/step_0050
if [ ! -d "$ADAPTER_GRPO" ]; then
  say "WARN: step_0050 missing, falling back to best.json"
  ADAPTER_GRPO=$("$PY" -c "import json;print(json.load(open('$ROOT/grpo_run/pilot_01/best.json'))['path'])" 2>/dev/null || echo "")
fi
if [ -n "$ADAPTER_GRPO" ] && [ -d "$ADAPTER_GRPO" ]; then
  say "start CSV #2 (grpo7b_step50) from $ADAPTER_GRPO"
  bash "$ROOT/scripts/run_submit.sh" "$ADAPTER_GRPO" grpo7b_step50 >> "$LOG" 2>&1
  say "CSV #2 rc=$?"
else
  say "FATAL: no GRPO checkpoint to score"
fi

say "chain finished"
