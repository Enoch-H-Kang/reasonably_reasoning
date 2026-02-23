#!/usr/bin/env python3
"""Aggregate multirun run-total CSVs into one combined mean table."""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import pandas as pd


REQUIRED_RUN_COLUMNS = {"mode", "game", "run", "seed", "total1", "total2", "total_score"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multirun run-total logs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="Input run_totals_*.csv files.",
    )
    parser.add_argument(
        "--inputs-from-file",
        default="",
        help="Optional text file with one run_totals_*.csv path per line.",
    )
    parser.add_argument(
        "--inputs-glob",
        default="",
        help="Optional glob for run_totals_*.csv files.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path for the combined table.")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> List[str]:
    paths: List[str] = list(args.inputs)
    if args.inputs_from_file:
        with open(args.inputs_from_file, "r", encoding="utf-8") as handle:
            for line in handle:
                path = line.strip()
                if path:
                    paths.append(path)
    if args.inputs_glob:
        paths.extend(sorted(glob.glob(args.inputs_glob)))
    paths = [os.path.abspath(path) for path in paths]
    if not paths:
        raise SystemExit("No input files were provided.")
    return sorted(set(paths))


def load_tables(paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_RUN_COLUMNS - set(frame.columns)
        if missing:
            raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
        frame = frame.copy()
        frame["source_file"] = os.path.abspath(path)
        frames.append(frame)
    if not frames:
        raise SystemExit("No input files provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args)
    run_totals = load_tables(paths)
    combined = (
        run_totals.groupby(["mode", "game"], as_index=False)
        .agg(
            runs=("run", "count"),
            mean_total_score=("total_score", "mean"),
            std_total_score=("total_score", "std"),
            min_total_score=("total_score", "min"),
            max_total_score=("total_score", "max"),
            mean_total1=("total1", "mean"),
            mean_total2=("total2", "mean"),
        )
        .sort_values(["mode", "game"])
        .reset_index(drop=True)
    )
    combined["std_total_score"] = combined["std_total_score"].fillna(0.0)

    out_parent = os.path.dirname(args.output)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print("Aggregated summary:")
    print(combined.to_string(index=False))
    print(f"\nSource run-total files: {len(paths)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
