#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./eval_stage_checkpoints.sh --run_dir <RUN_DIR> [options]

Required:
  --run_dir <path>          Run directory that contains nn/model_XXXX.pt
                            Example:
                            logs/rsl_rl/move_bottle_stage1/2026-03-17_10-00-00_move_bottle_stage1_v5

Optional:
  --task <name>             Task name (default: Isaac-E0509-Move-Bottle-Stage1-Direct-v0)
  --num_envs <n>            Number of eval envs (default: 16)
  --seed <n>                Fixed seed for fair comparison (default: 42)
  --from <iter>             First checkpoint iteration (default: 200)
  --to <iter>               Last checkpoint iteration (default: 20000)
  --step <iter>             Iteration step (default: 200)
  --video_length <steps>    Video length per checkpoint (default: 500)
  --headless                Run headless
  --dry_run                 Print commands only
  --help                    Show this help
EOF
}

RUN_DIR=""
TASK="Isaac-E0509-Move-Bottle-Stage1-Direct-v0"
NUM_ENVS=16
SEED=42
FROM_ITER=200
TO_ITER=20000
STEP_ITER=200
VIDEO_LENGTH=500
HEADLESS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="${2:-}"
      shift 2
      ;;
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    --num_envs)
      NUM_ENVS="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --from)
      FROM_ITER="${2:-}"
      shift 2
      ;;
    --to)
      TO_ITER="${2:-}"
      shift 2
      ;;
    --step)
      STEP_ITER="${2:-}"
      shift 2
      ;;
    --video_length)
      VIDEO_LENGTH="${2:-}"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "[ERROR] --run_dir is required."
  usage
  exit 1
fi

find_isaaclab_root() {
  local dir="$PWD"
  while [[ "$dir" != "/" ]]; do
    if [[ -x "$dir/isaaclab.sh" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

ROOT_DIR="$(find_isaaclab_root || true)"
if [[ -z "$ROOT_DIR" ]]; then
  echo "[ERROR] Could not find IsaacLab root (isaaclab.sh). Run from inside the IsaacLab repo."
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[ERROR] run_dir not found: $RUN_DIR"
  exit 1
fi

NN_DIR="$RUN_DIR/nn"
if [[ ! -d "$NN_DIR" ]]; then
  echo "[ERROR] nn directory not found in run_dir: $NN_DIR"
  exit 1
fi

echo "[INFO] Root       : $ROOT_DIR"
echo "[INFO] Run dir    : $RUN_DIR"
echo "[INFO] Task       : $TASK"
echo "[INFO] Num envs   : $NUM_ENVS"
echo "[INFO] Seed       : $SEED"
echo "[INFO] Iter range : $FROM_ITER..$TO_ITER (step $STEP_ITER)"
echo "[INFO] Video len  : $VIDEO_LENGTH"

for iter in $(seq "$FROM_ITER" "$STEP_ITER" "$TO_ITER"); do
  ckpt="$NN_DIR/model_${iter}.pt"
  if [[ ! -f "$ckpt" ]]; then
    continue
  fi

  cmd=(
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py
    --task="$TASK"
    --num_envs="$NUM_ENVS"
    --seed="$SEED"
    --checkpoint="$ckpt"
    --video
    --video_length="$VIDEO_LENGTH"
  )
  if [[ "$HEADLESS" -eq 1 ]]; then
    cmd+=(--headless)
  fi

  echo
  echo "=============================================================="
  echo "[EVAL] checkpoint: model_${iter}.pt"
  echo "=============================================================="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf ' %q' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
done

echo
echo "[INFO] Evaluation sweep completed."
