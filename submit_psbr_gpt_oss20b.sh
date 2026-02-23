#!/usr/bin/env bash
set -euo pipefail

cd /work1/krishnamurthy/arvind/repeatedgames
COLLUSIVE_MODE="${COLLUSIVE_MODE:-0}"
STRATEGY_MEMORY_ROUNDS="${STRATEGY_MEMORY_ROUNDS:-0}"
jid=$(sbatch --parsable --export=ALL,COLLUSIVE_MODE="$COLLUSIVE_MODE",STRATEGY_MEMORY_ROUNDS="$STRATEGY_MEMORY_ROUNDS" job.psbr_gpt_oss20b.sbatch)
WORK_ROOT="${WORK:-/work1/krishnamurthy/arvind}"
run_out="$WORK_ROOT/repeatedgames_runs_psbr/${jid}/slurm_${jid}.out"
run_err="$WORK_ROOT/repeatedgames_runs_psbr/${jid}/slurm_${jid}.err"

echo "Submitted PS-BR job: $jid"
echo "COLLUSIVE_MODE=$COLLUSIVE_MODE STRATEGY_MEMORY_ROUNDS=$STRATEGY_MEMORY_ROUNDS"
echo "Check queue: squeue -j $jid"
echo "Tail runtime log: tail -f $run_out"
echo "Tail runtime err: tail -f $run_err"
