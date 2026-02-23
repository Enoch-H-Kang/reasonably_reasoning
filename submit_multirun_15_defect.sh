#!/usr/bin/env bash
set -euo pipefail

cd /work1/krishnamurthy/arvind/repeatedgames
WORK_ROOT="${WORK:-/work1/krishnamurthy/arvind}"
PARTITION="${PARTITION:-mi2104x}"

RUNS_TOTAL="${RUNS_TOTAL:-100}"
RUNS_PER_JOB="${RUNS_PER_JOB:-25}"
ROUNDS="${ROUNDS:-100}"
FIRST_ACTION_MODE="${FIRST_ACTION_MODE:-defect}"
GAMES_PER_MODE="${GAMES_PER_MODE:-5}"
REQUIRE_IDLE_NODES="${REQUIRE_IDLE_NODES:-0}"
COLLUSIVE_MODE="${COLLUSIVE_MODE:-0}"

if (( RUNS_TOTAL % RUNS_PER_JOB != 0 )); then
  echo "RUNS_TOTAL ($RUNS_TOTAL) must be divisible by RUNS_PER_JOB ($RUNS_PER_JOB)." >&2
  exit 1
fi

CHUNKS=$(( RUNS_TOTAL / RUNS_PER_JOB ))
TOTAL_JOBS=$(( CHUNKS * 3 ))

if command -v sinfo >/dev/null 2>&1; then
  idle_nodes=$(sinfo -h -p "$PARTITION" -t idle -N | wc -l | tr -d ' ')
  alloc_nodes=$(sinfo -h -p "$PARTITION" -t alloc -N | wc -l | tr -d ' ')
  down_nodes=$(sinfo -h -p "$PARTITION" -t down -N | wc -l | tr -d ' ')
  echo "Partition $PARTITION node status: idle=$idle_nodes alloc=$alloc_nodes down=$down_nodes"
  echo "Planned submissions: $TOTAL_JOBS jobs ($CHUNKS chunks x 3 modes)"
  if (( idle_nodes == 0 )); then
    echo "No idle nodes are currently available on partition $PARTITION." >&2
    exit 1
  fi
  if (( REQUIRE_IDLE_NODES == 1 && idle_nodes < TOTAL_JOBS )); then
    echo "Insufficient idle nodes for strict launch policy (need $TOTAL_JOBS, have $idle_nodes)." >&2
    echo "Either wait for more idle nodes or set REQUIRE_IDLE_NODES=0 to allow queueing." >&2
    exit 1
  fi
else
  echo "Warning: sinfo not found; skipping pre-submit node availability check."
fi

mkdir -p "$WORK_ROOT/repeatedgames_runs_multirun"
manifest="$WORK_ROOT/repeatedgames_runs_multirun/manifest_multirun_$(date +%Y%m%d_%H%M%S).txt"
: > "$manifest"

declare -a base_jids=()
declare -a scot_jids=()
declare -a psbr_jids=()

for chunk in $(seq 0 $((CHUNKS - 1))); do
  idx=$((chunk + 1))
  tag=$(printf "base_r%s_n%s_chunk%02d_%s" "$ROUNDS" "$RUNS_PER_JOB" "$idx" "$FIRST_ACTION_MODE")
  seed_start=$((0 + chunk * RUNS_PER_JOB * GAMES_PER_MODE))
  jid=$(sbatch --parsable \
    --export=ALL,MODE=base,RUNS="$RUNS_PER_JOB",ROUNDS="$ROUNDS",SEED_START="$seed_start",FIRST_ACTION_MODE="$FIRST_ACTION_MODE",COLLUSIVE_MODE="$COLLUSIVE_MODE",OUT_TAG="$tag" \
    job.multirun_suite_gpt_oss20b.sbatch)
  base_jids+=("$jid")
  echo "$WORK_ROOT/repeatedgames_runs_multirun/${jid}_${tag}/run_totals_base_r${ROUNDS}_n${RUNS_PER_JOB}.csv" >> "$manifest"
  echo "Submitted BASE chunk $idx/$CHUNKS: $jid"
done

for chunk in $(seq 0 $((CHUNKS - 1))); do
  idx=$((chunk + 1))
  tag=$(printf "scot_r%s_n%s_chunk%02d_%s" "$ROUNDS" "$RUNS_PER_JOB" "$idx" "$FIRST_ACTION_MODE")
  seed_start=$((100000 + chunk * RUNS_PER_JOB * GAMES_PER_MODE))
  jid=$(sbatch --parsable \
    --export=ALL,MODE=scot,RUNS="$RUNS_PER_JOB",ROUNDS="$ROUNDS",SEED_START="$seed_start",FIRST_ACTION_MODE="$FIRST_ACTION_MODE",COLLUSIVE_MODE="$COLLUSIVE_MODE",OUT_TAG="$tag" \
    job.multirun_suite_gpt_oss20b.sbatch)
  scot_jids+=("$jid")
  echo "$WORK_ROOT/repeatedgames_runs_multirun/${jid}_${tag}/run_totals_scot_r${ROUNDS}_n${RUNS_PER_JOB}.csv" >> "$manifest"
  echo "Submitted SCOT chunk $idx/$CHUNKS: $jid"
done

for chunk in $(seq 0 $((CHUNKS - 1))); do
  idx=$((chunk + 1))
  tag=$(printf "psbr_r%sh20p5_n%s_chunk%02d_%s" "$ROUNDS" "$RUNS_PER_JOB" "$idx" "$FIRST_ACTION_MODE")
  seed_start=$((200000 + chunk * RUNS_PER_JOB * GAMES_PER_MODE))
  jid=$(sbatch --parsable \
    --export=ALL,MODE=psbr,RUNS="$RUNS_PER_JOB",ROUNDS="$ROUNDS",SEED_START="$seed_start",FIRST_ACTION_MODE="$FIRST_ACTION_MODE",COLLUSIVE_MODE="$COLLUSIVE_MODE",PS_SAMPLES=5,PLANNING_HORIZON=20,DISCOUNT=1.0,SAMPLE_TEMPERATURE=0.3,STRATEGY_INFERENCE=llm-label,OUT_TAG="$tag" \
    job.multirun_suite_gpt_oss20b.sbatch)
  psbr_jids+=("$jid")
  echo "$WORK_ROOT/repeatedgames_runs_multirun/${jid}_${tag}/run_totals_psbr_r${ROUNDS}_n${RUNS_PER_JOB}.csv" >> "$manifest"
  echo "Submitted PSBR chunk $idx/$CHUNKS: $jid"
done

echo
echo "Manifest of run-total logs:"
echo "  $manifest"

echo
echo "Queue checks:"
for jid in "${base_jids[@]}" "${scot_jids[@]}" "${psbr_jids[@]}"; do
  echo "  squeue -j $jid"
done

echo
echo "After completion, aggregate 15-row means with:"
echo "  python aggregate_multirun_means.py \\"
echo "    --inputs-from-file $manifest \\"
echo "    --output $WORK_ROOT/repeatedgames_runs_multirun/combined_15_mean_totals_$(date +%Y%m%d_%H%M%S).csv"
