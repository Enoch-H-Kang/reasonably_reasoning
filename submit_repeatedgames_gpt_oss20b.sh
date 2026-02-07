#!/usr/bin/env bash
set -euo pipefail

cd /work1/krishnamurthy/arvind/repeatedgames
jid=$(sbatch --parsable job.repeatedgames_gpt_oss20b.sbatch)
echo "Submitted job: $jid"
echo "Check queue: squeue -j $jid"
echo "Tail log:   tail -f repgames_oss20b.${jid}.out"
