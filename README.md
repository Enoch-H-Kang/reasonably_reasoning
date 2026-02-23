# repeatedgames (current fork)

This fork is focused on running repeated-game experiments with `openai/gpt-oss-20b` in three modes:

- base (`run_gpt_oss_games.py`)
- SCoT (`run_scot_games.py`)
- PS-BR (`run_ps_br_games.py`)

It also includes multi-run orchestration and aggregation tooling for large SLURM batches.

## Scope of this fork

Active code paths in this fork are centered on:

- `run_gpt_oss_games.py`
- `run_scot_games.py`
- `run_ps_br_games.py`
- `run_multirun_suite.py`
- `run_psbr_bos_multirun.py`
- `submit_*.sh` and `job.*.sbatch` launch scripts
- `aggregate_multirun_means.py` and `aggregate_round_window_means.py`

Legacy upstream assets that are not part of the active workflow were quarantined under:

- `quarantine/2026-02-23_reachability_candidates/`

## Supported games

All three main runners support:

- `bos` (Battle of the Sexes)
- `pd` (Prisoner's Dilemma)
- `deadlock`
- `samaritan`
- `lemons`
- `both` (alias for `bos` + `pd`)
- `all` (all five games)

## Environment setup (local)

```bash
cd /work1/krishnamurthy/arvind/repeatedgames
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is a baseline. On ROCm/HPC, the SLURM scripts install ROCm-specific torch explicitly.
- For `hf-local`, install a torch build compatible with your accelerator.

## Quick start (local)

### 1) Base mode

```bash
python run_gpt_oss_games.py \
  --backend hf-local \
  --game both \
  --model openai/gpt-oss-20b \
  --rounds 10
```

Default outputs:

- `bos/experiment_bos_gpt_oss_20b.csv`
- `pd/experiment_pd_gpt_oss_20b.csv`

### 2) SCoT mode

```bash
python run_scot_games.py \
  --backend hf-local \
  --game both \
  --model openai/gpt-oss-20b \
  --rounds 10
```

Default outputs:

- `scot/experiment_bos_scot_gpt_oss_20b.csv`
- `scot/experiment_pd_scot_gpt_oss_20b.csv`

### 3) PS-BR mode

```bash
python run_ps_br_games.py \
  --backend hf-local \
  --game both \
  --model openai/gpt-oss-20b \
  --rounds 50 \
  --ps-samples 4 \
  --planning-horizon 10 \
  --discount 1.0 \
  --sample-temperature 0.3
```

Default outputs:

- `ps_br/experiment_bos_psbr_gpt_oss_20b.csv`
- `ps_br/experiment_pd_psbr_gpt_oss_20b.csv`

PS-BR-specific controls include:

- `--strategy-inference {llm-label,likelihood}`
- `--strategy-memory-rounds <int>`
- `--collusive-mode`
- `--first-action-mode {model,defect}`

### Backend options

- Base: `--backend {hf-local,openai-compatible}`
- SCoT: `--backend {hf-local,openai-compatible,mock}`
- PS-BR: `--backend {hf-local,openai-compatible,mock}`

For `openai-compatible`, provide `--base-url` and `--api-key` (or set env vars like `OPENAI_BASE_URL`, `OPENAI_API_KEY`).

## CSV schema (high level)

- Base outputs include per-round actions and cumulative totals.
- SCoT outputs add `prediction1` and `prediction2` columns.
- PS-BR outputs add planner diagnostics, including selected strategies and estimated values.

## SLURM workflow (AMD/ROCm cluster)

Helper submit scripts:

- `./submit_repeatedgames_gpt_oss20b.sh`
- `./submit_scot_gpt_oss20b.sh`
- `./submit_psbr_gpt_oss20b.sh`
- `./submit_parallel_gpt_oss20b.sh` (base + scot + psbr)

Runtime logs are redirected under `$WORK` (default `/work1/krishnamurthy/arvind`), e.g.:

- base: `$WORK/repeatedgames_runs/<jobid>/slurm_<jobid>.out`
- scot: `$WORK/repeatedgames_runs_scot/<jobid>/slurm_<jobid>.out`
- psbr: `$WORK/repeatedgames_runs_psbr/<jobid>/slurm_<jobid>.out`

Useful PS-BR submit env knobs:

```bash
COLLUSIVE_MODE=1 STRATEGY_MEMORY_ROUNDS=10 ./submit_psbr_gpt_oss20b.sh
```

## Multi-run experiments

### Main multirun suite (all 5 games)

Driver script:

- `run_multirun_suite.py`

SLURM job wrapper:

- `job.multirun_suite_gpt_oss20b.sbatch`

Chunked launcher (submits base/scot/psbr batches and creates a manifest):

```bash
RUNS_TOTAL=100 RUNS_PER_JOB=25 ROUNDS=100 FIRST_ACTION_MODE=defect ./submit_multirun_15_defect.sh
```

Compatibility alias:

```bash
./submit_multirun_12_defect.sh
```

Per job, the suite writes:

- `round_logs_<mode>_r<rounds>_n<runs>.csv`
- `run_totals_<mode>_r<rounds>_n<runs>.csv`
- `summary_means_<mode>_r<rounds>_n<runs>.csv`
- `config_<mode>_r<rounds>_n<runs>.json`
- `progress_<mode>_r<rounds>_n<runs>.json`

And copies summary/totals CSVs to:

- `multirun_summaries/`

### BoS-only PS-BR multirun

- runner: `run_psbr_bos_multirun.py`
- sbatch: `job.psbr_bos_multirun_temp.sbatch`

## Aggregation utilities

### Aggregate run totals across many jobs

```bash
python aggregate_multirun_means.py \
  --inputs-from-file /path/to/manifest.txt \
  --output /path/to/combined_mean_totals.csv
```

Alternative input modes:

- `--inputs <file1> <file2> ...`
- `--inputs-glob '/path/**/run_totals_*.csv'`

### Aggregate windowed per-round rewards

```bash
python aggregate_round_window_means.py \
  --inputs-from-file /path/to/manifest.txt \
  --round-start 130 \
  --round-end 180 \
  --expected-runs 100 \
  --output /path/to/window_mean_rewards.csv
```

This utility maps each `run_totals_*.csv` to its sibling `round_logs_*.csv` and computes windowed reward summaries.

## Current repository layout

- `run_gpt_oss_games.py`: base repeated-game runner
- `run_scot_games.py`: SCoT runner
- `run_ps_br_games.py`: PS-BR runner
- `run_multirun_suite.py`: multi-run suite over all games
- `run_psbr_bos_multirun.py`: PS-BR BoS-only multirun
- `job.*.sbatch`: SLURM job definitions
- `submit_*.sh`: SLURM submit wrappers
- `aggregate_*.py`: postprocessing utilities
- `bos/`, `pd/`, `scot/`, `ps_br/`: output/data folders used by active workflows
- `multirun_summaries/`: copied summary and run-total outputs
- `quarantine/2026-02-23_reachability_candidates/`: archived non-active legacy content

## Repro tips

- Pin `--seed` for single runs and `--seed-start` for multi-run jobs.
- Keep `--first-action-mode` explicit in logs and job tags.
- For long multirun jobs, keep `--resume` enabled (already set in `job.multirun_suite_gpt_oss20b.sbatch`).
