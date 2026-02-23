#!/usr/bin/env python3
"""Aggregate per-setting mean window rewards from multirun round logs.

For each (mode, game, seed) run, this script sums (points1 + points2)
over an inclusive round window [round_start, round_end], then computes
summary stats across runs for each (mode, game).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


REQUIRED_COLUMNS = {"mode", "game", "seed", "round", "points1", "points2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate windowed round-score means from multirun outputs.")
    parser.add_argument(
        "--inputs-from-file",
        required=True,
        help="Manifest file containing run_totals_*.csv paths (one per line).",
    )
    parser.add_argument(
        "--round-start",
        type=int,
        default=130,
        help="Inclusive start round for the window.",
    )
    parser.add_argument(
        "--round-end",
        type=int,
        default=180,
        help="Inclusive end round for the window.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=100,
        help="Expected number of runs for each (mode, game).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the aggregated window means.",
    )
    return parser.parse_args()


def read_manifest(path: str) -> List[str]:
    paths: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            candidate = line.strip()
            if candidate:
                paths.append(os.path.abspath(candidate))
    if not paths:
        raise SystemExit(f"No input paths found in manifest: {path}")
    return paths


def derive_round_log_path(run_totals_path: str) -> str:
    parent = os.path.dirname(run_totals_path)
    basename = os.path.basename(run_totals_path)
    if not basename.startswith("run_totals_"):
        raise SystemExit(f"Unexpected run_totals filename: {run_totals_path}")
    round_name = basename.replace("run_totals_", "round_logs_", 1)
    round_path = os.path.join(parent, round_name)
    if not os.path.exists(round_path):
        raise SystemExit(f"Missing round log file for {run_totals_path}: {round_path}")
    return round_path


def load_round_logs(round_log_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in round_log_paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
        frame = frame.copy()
        frame["source_file"] = os.path.abspath(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    if args.round_start > args.round_end:
        raise SystemExit("--round-start must be <= --round-end")

    run_totals_paths = read_manifest(args.inputs_from_file)
    round_log_paths = [derive_round_log_path(path) for path in run_totals_paths]
    logs = load_round_logs(round_log_paths)

    window = logs[(logs["round"] >= args.round_start) & (logs["round"] <= args.round_end)].copy()
    if window.empty:
        raise SystemExit("No rows in selected round window.")

    window["round_sum_reward"] = window["points1"] + window["points2"]

    expected_window_len = args.round_end - args.round_start + 1
    per_run_counts = (
        window.groupby(["mode", "game", "seed"], as_index=False)
        .agg(window_rounds=("round", "count"))
    )
    bad_counts = per_run_counts[per_run_counts["window_rounds"] != expected_window_len]
    if not bad_counts.empty:
        raise SystemExit(
            "Some runs are missing rounds in the requested window. "
            f"Expected {expected_window_len} rounds per run."
        )

    per_run = (
        window.groupby(["mode", "game", "seed"], as_index=False)
        .agg(window_sum_reward=("round_sum_reward", "sum"))
    )

    summary = (
        per_run.groupby(["mode", "game"], as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_window_sum_reward=("window_sum_reward", "mean"),
            std_window_sum_reward=("window_sum_reward", "std"),
            min_window_sum_reward=("window_sum_reward", "min"),
            max_window_sum_reward=("window_sum_reward", "max"),
        )
        .sort_values(["mode", "game"])
        .reset_index(drop=True)
    )
    summary["std_window_sum_reward"] = summary["std_window_sum_reward"].fillna(0.0)

    if args.expected_runs > 0:
        bad_run_counts = summary[summary["runs"] != args.expected_runs]
        if not bad_run_counts.empty:
            raise SystemExit(
                "Run-count mismatch for some settings. "
                f"Expected {args.expected_runs} runs each.\n"
                f"{bad_run_counts.to_string(index=False)}"
            )

    out_parent = os.path.dirname(args.output)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("Window-reward summary (per mode/game):")
    print(summary.to_string(index=False))
    print(f"\nWindow: rounds {args.round_start}..{args.round_end} (inclusive)")
    print(f"Sources: {len(round_log_paths)} round-log files")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
