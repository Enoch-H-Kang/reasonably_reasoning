#!/usr/bin/env python3
"""Run PS-BR multi-run for BoS only and aggregate run-level totals."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import pandas as pd

import run_ps_br_games as psbr


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PS-BR BoS-only multi-run suite.")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name.")
    parser.add_argument("--runs", type=int, default=100, help="Number of independent runs.")
    parser.add_argument("--rounds", type=int, default=100, help="Rounds per run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Seed offset for run 1.")
    parser.add_argument(
        "--first-action-mode",
        choices=["model", "defect"],
        default="model",
        help="Round-1 action mode.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for logs/CSVs.")

    parser.add_argument("--device-map", default="cuda", help="Transformers device map.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Max generated tokens per call.")
    parser.add_argument("--parse-retries", type=int, default=2, help="Parse retries.")
    parser.add_argument(
        "--mxfp4-dequantize",
        action="store_true",
        help="Force MXFP4 dequantization during model load.",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=["auto", "batched_mm", "grouped_mm", "eager"],
        default="eager",
        help="MoE expert implementation.",
    )

    parser.add_argument("--ps-samples", type=int, default=5, help="Number of rollout samples.")
    parser.add_argument("--planning-horizon", type=int, default=20, help="Lookahead horizon.")
    parser.add_argument("--discount", type=float, default=1.0, help="Discount factor.")
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3).",
    )
    parser.add_argument(
        "--strategy-inference",
        choices=["llm-label", "likelihood"],
        default="llm-label",
        help="Opponent strategy inference mode.",
    )
    parser.add_argument(
        "--strategy-memory-rounds",
        type=int,
        default=0,
        help="Opponent strategy inference memory window. 0 means full history.",
    )
    parser.add_argument(
        "--collusive-mode",
        action="store_true",
        help="Enable collusive prior guidance for PS-BR strategy-label inference.",
    )
    return parser.parse_args()


def run_suite(args: argparse.Namespace) -> Dict[str, str]:
    ensure_dir(args.output_dir)

    if args.ps_samples <= 0:
        raise SystemExit("--ps-samples must be >= 1")
    if not (0.0 <= args.discount <= 1.0):
        raise SystemExit("--discount must be in [0, 1]")
    if args.strategy_memory_rounds < 0:
        raise SystemExit("--strategy-memory-rounds must be >= 0")

    backend = psbr.HuggingFaceLocalBackend(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        mxfp4_dequantize=args.mxfp4_dequantize,
        experts_implementation=args.experts_implementation,
        parse_retries=args.parse_retries,
    )
    planner_cfg = psbr.PlannerConfig(
        samples=args.ps_samples,
        planning_horizon=args.planning_horizon,
        discount=args.discount,
        sample_temperature=args.sample_temperature,
        strategy_inference=args.strategy_inference,
        strategy_memory_rounds=args.strategy_memory_rounds,
        collusive_mode=args.collusive_mode,
    )

    detail_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, float]] = []
    wall_start = time.time()

    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed_start + run_idx - 1
        random.seed(run_seed)

        t0 = time.time()
        frame = psbr.run_bos_psbr(
            backend_1=backend,
            backend_2=backend,
            model_name=args.model,
            rounds=args.rounds,
            planner_cfg=planner_cfg,
            first_action_mode=args.first_action_mode,
        ).copy()

        frame["psbr_samples"] = planner_cfg.samples
        frame["psbr_horizon"] = planner_cfg.planning_horizon
        frame["psbr_discount"] = planner_cfg.discount
        frame["psbr_sample_temperature"] = planner_cfg.sample_temperature
        frame["psbr_strategy_inference"] = planner_cfg.strategy_inference
        frame["psbr_strategy_memory_rounds"] = planner_cfg.strategy_memory_rounds
        frame["psbr_collusive_mode"] = planner_cfg.collusive_mode

        final_row = frame.iloc[-1]
        final_total1 = float(final_row["total1"])
        final_total2 = float(final_row["total2"])
        total_score = final_total1 + final_total2

        frame.insert(0, "final_total_score", total_score)
        frame.insert(0, "seed", run_seed)
        frame.insert(0, "run", run_idx)
        frame.insert(0, "game", "bos")
        frame.insert(0, "mode", "psbr")
        detail_frames.append(frame)

        run_rows.append(
            {
                "mode": "psbr",
                "game": "bos",
                "run": run_idx,
                "seed": run_seed,
                "total_score": total_score,
                "total1": final_total1,
                "total2": final_total2,
            }
        )

        print(
            f"[{run_idx}/{args.runs}] mode=psbr game=bos run={run_idx} seed={run_seed} "
            f"total_score={total_score:.2f} elapsed={time.time() - t0:.2f}s",
            flush=True,
        )

    detail_df = pd.concat(detail_frames, ignore_index=True)
    run_totals_df = pd.DataFrame(run_rows)
    summary_df = (
        run_totals_df.groupby(["mode", "game"], as_index=False)
        .agg(
            runs=("total_score", "size"),
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

    suffix = f"psbr_bos_r{args.rounds}_n{args.runs}_temp{args.sample_temperature:g}"
    detail_path = os.path.join(args.output_dir, f"round_logs_{suffix}.csv")
    totals_path = os.path.join(args.output_dir, f"run_totals_{suffix}.csv")
    summary_path = os.path.join(args.output_dir, f"summary_means_{suffix}.csv")
    config_path = os.path.join(args.output_dir, f"config_{suffix}.json")

    detail_df.to_csv(detail_path, index=False)
    run_totals_df.to_csv(totals_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "psbr",
                "game": "bos",
                "model": args.model,
                "runs": args.runs,
                "rounds": args.rounds,
                "seed_start": args.seed_start,
                "first_action_mode": args.first_action_mode,
                "ps_samples": args.ps_samples,
                "planning_horizon": args.planning_horizon,
                "discount": args.discount,
                "sample_temperature": args.sample_temperature,
                "strategy_inference": args.strategy_inference,
                "strategy_memory_rounds": args.strategy_memory_rounds,
                "collusive_mode": args.collusive_mode,
                "device_map": args.device_map,
                "torch_dtype": args.torch_dtype,
                "max_new_tokens": args.max_new_tokens,
                "parse_retries": args.parse_retries,
                "mxfp4_dequantize": args.mxfp4_dequantize,
                "experts_implementation": args.experts_implementation,
                "wall_time_sec": time.time() - wall_start,
            },
            f,
            indent=2,
        )

    print("Wrote outputs:", flush=True)
    print(f"  {detail_path}", flush=True)
    print(f"  {totals_path}", flush=True)
    print(f"  {summary_path}", flush=True)
    print(f"  {config_path}", flush=True)
    return {
        "detail_path": detail_path,
        "run_totals_path": totals_path,
        "summary_path": summary_path,
        "config_path": config_path,
    }


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
