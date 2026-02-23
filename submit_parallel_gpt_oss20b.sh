#!/usr/bin/env bash
set -euo pipefail

cd /work1/krishnamurthy/arvind/repeatedgames
WORK_ROOT="${WORK:-/work1/krishnamurthy/arvind}"

jid_base=$(sbatch --parsable job.repeatedgames_gpt_oss20b.sbatch)
jid_scot=$(sbatch --parsable job.scot_gpt_oss20b.sbatch)
jid_psbr=$(sbatch --parsable job.psbr_gpt_oss20b.sbatch)

echo "Submitted BASE job: $jid_base"
echo "Submitted SCOT job: $jid_scot"
echo "Submitted PSBR job: $jid_psbr"

echo "Queue checks:"
echo "  squeue -j $jid_base"
echo "  squeue -j $jid_scot"
echo "  squeue -j $jid_psbr"

echo "Runtime logs:"
echo "  tail -f $WORK_ROOT/repeatedgames_runs/${jid_base}/slurm_${jid_base}.out"
echo "  tail -f $WORK_ROOT/repeatedgames_runs/${jid_base}/slurm_${jid_base}.err"
echo "  tail -f $WORK_ROOT/repeatedgames_runs_scot/${jid_scot}/slurm_${jid_scot}.out"
echo "  tail -f $WORK_ROOT/repeatedgames_runs_scot/${jid_scot}/slurm_${jid_scot}.err"
echo "  tail -f $WORK_ROOT/repeatedgames_runs_psbr/${jid_psbr}/slurm_${jid_psbr}.out"
echo "  tail -f $WORK_ROOT/repeatedgames_runs_psbr/${jid_psbr}/slurm_${jid_psbr}.err"
