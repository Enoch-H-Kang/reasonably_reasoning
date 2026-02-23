#!/usr/bin/env bash
set -euo pipefail

cd /work1/krishnamurthy/arvind/repeatedgames
jid=$(sbatch --parsable job.scot_gpt_oss20b.sbatch)
WORK_ROOT="${WORK:-/work1/krishnamurthy/arvind}"
run_out="$WORK_ROOT/repeatedgames_runs_scot/${jid}/slurm_${jid}.out"
run_err="$WORK_ROOT/repeatedgames_runs_scot/${jid}/slurm_${jid}.err"

echo "Submitted SCoT job: $jid"
echo "Check queue: squeue -j $jid"
echo "Tail runtime log: tail -f $run_out"
echo "Tail runtime err: tail -f $run_err"
