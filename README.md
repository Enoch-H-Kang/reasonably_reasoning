# repeatedgames

Repeated-game self-play experiments with `openai/gpt-oss-20b`.

This repo provides three decision procedures:

- Base (`run_gpt_oss_games.py`)
- SCoT (`run_scot_games.py`)
- PS-BR (`run_ps_br_games.py`)

Supported games:

- `bos`
- `pd`
- `deadlock`
- `samaritan`
- `lemons`
- `both` (`bos` + `pd`)
- `all` (all five games)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Single-run usage

### Base

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

### SCoT

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

### PS-BR

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

Useful PS-BR options:

- `--strategy-inference {llm-label,likelihood}`
- `--strategy-memory-rounds <int>`
- `--collusive-mode`
- `--first-action-mode {model,defect}`

## Backends

- Base: `--backend {hf-local,openai-compatible}`
- SCoT: `--backend {hf-local,openai-compatible,mock}`
- PS-BR: `--backend {hf-local,openai-compatible,mock}`

For `openai-compatible`, use `--base-url` and `--api-key` (or environment variables such as `OPENAI_BASE_URL`, `OPENAI_API_KEY`).

## SLURM scripts

Single jobs:

- `./submit_repeatedgames_gpt_oss20b.sh`
- `./submit_scot_gpt_oss20b.sh`
- `./submit_psbr_gpt_oss20b.sh`
- `./submit_parallel_gpt_oss20b.sh`

Multirun:

- `run_multirun_suite.py` (direct driver)
- `job.multirun_suite_gpt_oss20b.sbatch` (batch wrapper)
- `./submit_multirun_15_defect.sh` (chunked launcher)
- `./submit_multirun_12_defect.sh` (alias)

Example chunked launch:

```bash
RUNS_TOTAL=100 RUNS_PER_JOB=25 ROUNDS=200 FIRST_ACTION_MODE=defect ./submit_multirun_15_defect.sh
```

Per job output files:

- `round_logs_<mode>_r<rounds>_n<runs>.csv`
- `run_totals_<mode>_r<rounds>_n<runs>.csv`
- `summary_means_<mode>_r<rounds>_n<runs>.csv`
- `config_<mode>_r<rounds>_n<runs>.json`
- `progress_<mode>_r<rounds>_n<runs>.json`

BoS-only PS-BR multirun:

- `run_psbr_bos_multirun.py`
- `job.psbr_bos_multirun_temp.sbatch`

## Aggregation

Aggregate run totals:

```bash
python aggregate_multirun_means.py \
  --inputs-from-file /path/to/manifest.txt \
  --output /path/to/combined_mean_totals.csv
```

Aggregate windowed round rewards:

```bash
python aggregate_round_window_means.py \
  --inputs-from-file /path/to/manifest.txt \
  --round-start 130 \
  --round-end 180 \
  --expected-runs 100 \
  --output /path/to/window_mean_rewards.csv
```
