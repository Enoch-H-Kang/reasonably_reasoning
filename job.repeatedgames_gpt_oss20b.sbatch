#!/bin/bash
#SBATCH -J repgames_oss20b
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p mi2104x
#SBATCH -t 12:00:00
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

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$WORK_ROOT/repeatedgames_runs/$RUN_ID"
mkdir -p "$OUT_ROOT"

# Put all logs in run folder.
exec > "$OUT_ROOT/slurm_${SLURM_JOB_ID}.out" 2> "$OUT_ROOT/slurm_${SLURM_JOB_ID}.err"

echo "OUT_ROOT=$OUT_ROOT"
echo "HOST=$(hostname)"
echo "START_TIME=$(date -Iseconds)"

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

python -m pip install -U pip setuptools wheel

# Install ROCm torch and local inference stack.
python -m pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1+rocm6.4
python -m pip install -U transformers kernels accelerate pandas openai

echo "==== rocm-smi (sanity) ===="
rocm-smi || true
echo "==========================="

# Use one GPU process for local HF inference.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

echo "Running repeated BOS + PD with openai/gpt-oss-20b ..."
python run_gpt_oss_games.py \
  --backend hf-local \
  --game both \
  --model openai/gpt-oss-20b \
  --rounds 10 \
  --device-map cuda \
  --torch-dtype bfloat16 \
  --mxfp4-dequantize \
  --experts-implementation eager \
  --max-new-tokens 4 \
  --bos-output "$OUT_ROOT/experiment_bos_gpt_oss_20b.csv" \
  --pd-output "$OUT_ROOT/experiment_pd_gpt_oss_20b.csv"

# Also publish latest outputs in repo-standard locations.
cp "$OUT_ROOT/experiment_bos_gpt_oss_20b.csv" "$REPO_DIR/bos/experiment_bos_gpt_oss_20b.csv"
cp "$OUT_ROOT/experiment_pd_gpt_oss_20b.csv" "$REPO_DIR/pd/experiment_pd_gpt_oss_20b.csv"

echo "END_TIME=$(date -Iseconds)"
echo "Done. Outputs:"
echo "  $OUT_ROOT/experiment_bos_gpt_oss_20b.csv"
echo "  $OUT_ROOT/experiment_pd_gpt_oss_20b.csv"
