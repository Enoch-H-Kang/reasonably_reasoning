#!/bin/bash
#SBATCH -J psbr_bos_temp
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p mi2104x
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16

set -euo pipefail

module purge
module load hpcfund
module load rocm/6.4.1
module list

WORK_ROOT="${WORK:-/work1/krishnamurthy/arvind}"
REPO_DIR="$WORK_ROOT/repeatedgames"
cd "$REPO_DIR"

RUNS="${RUNS:-100}"
ROUNDS="${ROUNDS:-100}"
SEED_START="${SEED_START:-500000}"
FIRST_ACTION_MODE="${FIRST_ACTION_MODE:-model}"

PS_SAMPLES="${PS_SAMPLES:-5}"
PLANNING_HORIZON="${PLANNING_HORIZON:-20}"
DISCOUNT="${DISCOUNT:-1.0}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.3}"
STRATEGY_INFERENCE="${STRATEGY_INFERENCE:-llm-label}"
STRATEGY_MEMORY_ROUNDS="${STRATEGY_MEMORY_ROUNDS:-0}"
COLLUSIVE_MODE="${COLLUSIVE_MODE:-0}"

TEMP_TAG="${TEMP_TAG:-$(echo "$SAMPLE_TEMPERATURE" | tr -d '.')}"
OUT_TAG="${OUT_TAG:-psbr_bos_r${ROUNDS}h${PLANNING_HORIZON}p${PS_SAMPLES}_n${RUNS}_${FIRST_ACTION_MODE}_temp${TEMP_TAG}}"
RUN_ID="${SLURM_JOB_ID}_${OUT_TAG}"
OUT_ROOT="$WORK_ROOT/repeatedgames_runs_multirun/$RUN_ID"
mkdir -p "$OUT_ROOT"

exec > "$OUT_ROOT/slurm_${SLURM_JOB_ID}.out" 2> "$OUT_ROOT/slurm_${SLURM_JOB_ID}.err"

echo "OUT_ROOT=$OUT_ROOT"
echo "HOST=$(hostname)"
echo "START_TIME=$(date -Iseconds)"
echo "PARAMS mode=psbr game=bos runs=$RUNS rounds=$ROUNDS seed_start=$SEED_START first_action_mode=$FIRST_ACTION_MODE"
echo "PARAMS ps_samples=$PS_SAMPLES planning_horizon=$PLANNING_HORIZON discount=$DISCOUNT sample_temperature=$SAMPLE_TEMPERATURE strategy_inference=$STRATEGY_INFERENCE strategy_memory_rounds=$STRATEGY_MEMORY_ROUNDS collusive_mode=$COLLUSIVE_MODE"

COLLUSIVE_FLAG=()
if [[ "$COLLUSIVE_MODE" == "1" ]]; then
  COLLUSIVE_FLAG+=(--collusive-mode)
fi

export HF_HOME="$WORK_ROOT/repeatedgames_cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORK_ROOT/repeatedgames_cache/xdg"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

VENV_DIR="$WORK_ROOT/venv/repeatedgames_oss20b"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

INSTALL_LOCK="$VENV_DIR/.pip_install.lock"
exec 9>"$INSTALL_LOCK"
flock 9
python -m pip install pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1+rocm6.4
python -m pip install transformers kernels accelerate pandas openai
flock -u 9

echo "==== rocm-smi (sanity) ===="
rocm-smi || true
echo "==========================="

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

python run_psbr_bos_multirun.py \
  --model openai/gpt-oss-20b \
  --runs "$RUNS" \
  --rounds "$ROUNDS" \
  --seed-start "$SEED_START" \
  --first-action-mode "$FIRST_ACTION_MODE" \
  --output-dir "$OUT_ROOT" \
  --device-map cuda \
  --torch-dtype bfloat16 \
  --mxfp4-dequantize \
  --experts-implementation eager \
  --max-new-tokens 4 \
  --parse-retries 2 \
  --ps-samples "$PS_SAMPLES" \
  --planning-horizon "$PLANNING_HORIZON" \
  --discount "$DISCOUNT" \
  --sample-temperature "$SAMPLE_TEMPERATURE" \
  --strategy-inference "$STRATEGY_INFERENCE" \
  --strategy-memory-rounds "$STRATEGY_MEMORY_ROUNDS" \
  "${COLLUSIVE_FLAG[@]}"

mkdir -p "$REPO_DIR/multirun_summaries"
cp "$OUT_ROOT"/summary_means_*.csv "$REPO_DIR/multirun_summaries/" || true
cp "$OUT_ROOT"/run_totals_*.csv "$REPO_DIR/multirun_summaries/" || true

echo "END_TIME=$(date -Iseconds)"
echo "Done. Outputs:"
ls -1 "$OUT_ROOT"
