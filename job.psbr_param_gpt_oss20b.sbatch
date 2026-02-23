#!/bin/bash
#SBATCH -J psbr_param_oss20b
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

ROUNDS="${ROUNDS:-50}"
PLANNING_HORIZON="${PLANNING_HORIZON:-10}"
PS_SAMPLES="${PS_SAMPLES:-1}"
DISCOUNT="${DISCOUNT:-1.0}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.3}"
FIRST_ACTION_MODE="${FIRST_ACTION_MODE:-model}"
OUT_TAG="${OUT_TAG:-r${ROUNDS}h${PLANNING_HORIZON}p${PS_SAMPLES}}"

# Use SLURM job id + explicit tag so each run has a deterministic output folder.
RUN_ID="${SLURM_JOB_ID}_${OUT_TAG}"
OUT_ROOT="$WORK_ROOT/repeatedgames_runs_psbr/$RUN_ID"
mkdir -p "$OUT_ROOT"

# Put all logs in run folder.
exec > "$OUT_ROOT/slurm_${SLURM_JOB_ID}.out" 2> "$OUT_ROOT/slurm_${SLURM_JOB_ID}.err"

echo "OUT_ROOT=$OUT_ROOT"
echo "HOST=$(hostname)"
echo "START_TIME=$(date -Iseconds)"
echo "PARAMS rounds=$ROUNDS horizon=$PLANNING_HORIZON ps_samples=$PS_SAMPLES discount=$DISCOUNT sample_temperature=$SAMPLE_TEMPERATURE first_action_mode=$FIRST_ACTION_MODE"

# Use work-backed cache for large model files.
export HF_HOME="$WORK_ROOT/repeatedgames_cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORK_ROOT/repeatedgames_cache/xdg"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

# Reusable venv for repeated games runs.
VENV_DIR="$WORK_ROOT/venv/repeatedgames_oss20b"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Prevent concurrent jobs from mutating the shared venv at the same time.
INSTALL_LOCK="$VENV_DIR/.pip_install.lock"
exec 9>"$INSTALL_LOCK"
flock 9
python -m pip install pip setuptools wheel

# Install ROCm torch and local inference stack.
python -m pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1+rocm6.4
python -m pip install transformers kernels accelerate pandas openai
flock -u 9

echo "==== rocm-smi (sanity) ===="
rocm-smi || true
echo "==========================="

# Use one GPU process for local HF inference.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

echo "Running PS-BR repeated BOS + PD with openai/gpt-oss-20b ..."
python run_ps_br_games.py \
  --backend hf-local \
  --game both \
  --model openai/gpt-oss-20b \
  --rounds "$ROUNDS" \
  --first-action-mode "$FIRST_ACTION_MODE" \
  --ps-samples "$PS_SAMPLES" \
  --planning-horizon "$PLANNING_HORIZON" \
  --discount "$DISCOUNT" \
  --sample-temperature "$SAMPLE_TEMPERATURE" \
  --device-map cuda \
  --torch-dtype bfloat16 \
  --mxfp4-dequantize \
  --experts-implementation eager \
  --max-new-tokens 4 \
  --bos-output "$OUT_ROOT/experiment_bos_psbr_gpt_oss_20b.csv" \
  --pd-output "$OUT_ROOT/experiment_pd_psbr_gpt_oss_20b.csv"

mkdir -p "$REPO_DIR/ps_br"
cp "$OUT_ROOT/experiment_bos_psbr_gpt_oss_20b.csv" "$REPO_DIR/ps_br/experiment_bos_psbr_gpt_oss_20b.csv"
cp "$OUT_ROOT/experiment_pd_psbr_gpt_oss_20b.csv" "$REPO_DIR/ps_br/experiment_pd_psbr_gpt_oss_20b.csv"

echo "END_TIME=$(date -Iseconds)"
echo "Done. Outputs:"
echo "  $OUT_ROOT/experiment_bos_psbr_gpt_oss_20b.csv"
echo "  $OUT_ROOT/experiment_pd_psbr_gpt_oss_20b.csv"
