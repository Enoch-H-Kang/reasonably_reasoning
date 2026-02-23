#!/usr/bin/env python3
"""Run multi-run repeated-games experiments and aggregate total-score means.

This script is designed for the 3x5 setup:
- modes: base, scot, psbr
- games: bos, pd, deadlock, samaritan, lemons

For each (mode, game), it runs N independent runs, logs per-round/per-run data,
and writes mean total score statistics where total score = total1 + total2.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Set, Tuple

import pandas as pd


GAMES: Tuple[str, ...] = ("bos", "pd", "deadlock", "samaritan", "lemons")
MODES: Tuple[str, ...] = ("base", "scot", "psbr")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def append_csv(path: str, frame: pd.DataFrame) -> None:
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    frame.to_csv(path, mode="a", header=header, index=False)


def canonicalize_run_totals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    dedup = frame.drop_duplicates(subset=["mode", "game", "run"], keep="last")
    return dedup.sort_values(["mode", "run", "game"]).reset_index(drop=True)


def canonicalize_round_logs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    dedup_cols = [col for col in ("mode", "game", "run", "seed", "round") if col in frame.columns]
    if not dedup_cols:
        return frame
    dedup = frame.drop_duplicates(subset=dedup_cols, keep="last")
    return dedup.sort_values(["mode", "run", "game", "round"]).reset_index(drop=True)


def task_seed(seed_start: int, run_idx: int, game_idx: int) -> int:
    return seed_start + (run_idx - 1) * len(GAMES) + game_idx


def load_completed_tasks(run_totals_path: str, resume: bool) -> Tuple[pd.DataFrame, Set[Tuple[int, str]]]:
    if not (resume and os.path.exists(run_totals_path)):
        return pd.DataFrame(), set()

    run_totals_df = pd.read_csv(run_totals_path)
    canonical = canonicalize_run_totals(run_totals_df)
    if len(canonical) != len(run_totals_df):
        print(
            f"Resume cleanup: removed {len(run_totals_df) - len(canonical)} duplicate run-total rows "
            f"from {run_totals_path}"
        )
        canonical.to_csv(run_totals_path, index=False)

    completed = {
        (int(row.run), str(row.game))
        for row in canonical.itertuples(index=False)
    }
    return canonical, completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-run repeated-games suite with aggregation.")
    parser.add_argument("--mode", choices=MODES, required=True, help="Experiment mode to run.")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name.")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per game.")
    parser.add_argument("--rounds", type=int, default=100, help="Rounds per run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Base seed used to derive per-(run,game) seeds.")
    parser.add_argument(
        "--first-action-mode",
        choices=["model", "defect"],
        default="defect",
        help="Round-1 action mode.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for all logs/CSVs.")

    # HF-local model loading args (experiment currently targets hf-local backend).
    parser.add_argument("--device-map", default="cuda", help="Transformers device map.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Max generated tokens per model call.")
    parser.add_argument("--parse-retries", type=int, default=2, help="Parser retries (scot/psbr only).")
    parser.add_argument(
        "--mxfp4-dequantize",
        action="store_true",
        help="Force MXFP4 dequantization to bf16 during model load.",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=["auto", "batched_mm", "grouped_mm", "eager"],
        default="eager",
        help="MoE expert implementation.",
    )

    # PS-BR planner args.
    parser.add_argument("--ps-samples", type=int, default=5, help="PS-BR sampled rollouts.")
    parser.add_argument("--planning-horizon", type=int, default=20, help="PS-BR lookahead horizon.")
    parser.add_argument("--discount", type=float, default=1.0, help="PS-BR discount factor.")
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help=(
            "PS-BR sampling temperature. "
            "Default is 0.3 for all games."
        ),
    )
    parser.add_argument(
        "--strategy-inference",
        choices=["llm-label", "likelihood"],
        default="llm-label",
        help="PS-BR opponent strategy inference mode.",
    )
    parser.add_argument(
        "--strategy-memory-rounds",
        type=int,
        default=0,
        help="PS-BR opponent strategy inference memory window. 0 means full history.",
    )
    parser.add_argument(
        "--collusive-mode",
        action="store_true",
        help="Enable collusive prior guidance for PS-BR strategy-label inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run_totals/round_logs files in --output-dir.",
    )

    return parser.parse_args()


def build_runner(mode: str, args: argparse.Namespace):
    if mode == "base":
        import run_gpt_oss_games as mod

        backend_1 = mod.HuggingFaceLocalBackend(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            mxfp4_dequantize=args.mxfp4_dequantize,
            experts_implementation=args.experts_implementation,
        )
        backend_2 = backend_1

        def run_one(game: str) -> pd.DataFrame:
            if game == "bos":
                return mod.run_bos(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "pd":
                return mod.run_pd(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "deadlock":
                return mod.run_deadlock(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "samaritan":
                return mod.run_samaritan(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "lemons":
                return mod.run_lemons(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            raise ValueError(f"Unsupported game: {game}")

        return run_one

    if mode == "scot":
        import run_scot_games as mod

        backend_1 = mod.HuggingFaceLocalBackend(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            mxfp4_dequantize=args.mxfp4_dequantize,
            experts_implementation=args.experts_implementation,
            parse_retries=args.parse_retries,
        )
        backend_2 = backend_1

        def run_one(game: str) -> pd.DataFrame:
            if game == "bos":
                return mod.run_bos_scot(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "pd":
                return mod.run_pd_scot(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "deadlock":
                return mod.run_deadlock_scot(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "samaritan":
                return mod.run_samaritan_scot(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            if game == "lemons":
                return mod.run_lemons_scot(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    first_action_mode=args.first_action_mode,
                )
            raise ValueError(f"Unsupported game: {game}")

        return run_one

    if mode == "psbr":
        import run_ps_br_games as mod

        if args.ps_samples <= 0:
            raise SystemExit("--ps-samples must be >= 1")
        if not (0.0 <= args.discount <= 1.0):
            raise SystemExit("--discount must be in [0, 1]")
        if args.strategy_memory_rounds < 0:
            raise SystemExit("--strategy-memory-rounds must be >= 0")

        backend_1 = mod.HuggingFaceLocalBackend(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            mxfp4_dequantize=args.mxfp4_dequantize,
            experts_implementation=args.experts_implementation,
            parse_retries=args.parse_retries,
        )
        backend_2 = backend_1

        def resolve_sample_temperature(_game: str) -> float:
            if args.sample_temperature is not None:
                return args.sample_temperature
            return 0.3

        def run_one(game: str) -> pd.DataFrame:
            planner_cfg = mod.PlannerConfig(
                samples=args.ps_samples,
                planning_horizon=args.planning_horizon,
                discount=args.discount,
                sample_temperature=resolve_sample_temperature(game),
                strategy_inference=args.strategy_inference,
                strategy_memory_rounds=args.strategy_memory_rounds,
                collusive_mode=args.collusive_mode,
            )
            if game == "bos":
                frame = mod.run_bos_psbr(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    planner_cfg,
                    first_action_mode=args.first_action_mode,
                )
            elif game == "pd":
                frame = mod.run_pd_psbr(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    planner_cfg,
                    first_action_mode=args.first_action_mode,
                )
            elif game == "deadlock":
                frame = mod.run_deadlock_psbr(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    planner_cfg,
                    first_action_mode=args.first_action_mode,
                )
            elif game == "samaritan":
                frame = mod.run_samaritan_psbr(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    planner_cfg,
                    first_action_mode=args.first_action_mode,
                )
            elif game == "lemons":
                frame = mod.run_lemons_psbr(
                    backend_1,
                    backend_2,
                    args.model,
                    args.rounds,
                    planner_cfg,
                    first_action_mode=args.first_action_mode,
                )
            else:
                raise ValueError(f"Unsupported game: {game}")

            # Keep planner metadata in the detail logs as well.
            frame["psbr_samples"] = planner_cfg.samples
            frame["psbr_horizon"] = planner_cfg.planning_horizon
            frame["psbr_discount"] = planner_cfg.discount
            frame["psbr_sample_temperature"] = planner_cfg.sample_temperature
            frame["psbr_strategy_inference"] = planner_cfg.strategy_inference
            frame["psbr_strategy_memory_rounds"] = planner_cfg.strategy_memory_rounds
            frame["psbr_collusive_mode"] = planner_cfg.collusive_mode
            return frame

        return run_one

    raise ValueError(f"Unsupported mode: {mode}")


def run_suite(args: argparse.Namespace) -> Dict[str, str]:
    ensure_dir(args.output_dir)
    run_one = build_runner(args.mode, args)

    prefix = f"{args.mode}_r{args.rounds}_n{args.runs}"
    detail_path = os.path.join(args.output_dir, f"round_logs_{prefix}.csv")
    run_totals_path = os.path.join(args.output_dir, f"run_totals_{prefix}.csv")
    summary_path = os.path.join(args.output_dir, f"summary_means_{prefix}.csv")
    config_path = os.path.join(args.output_dir, f"config_{prefix}.json")
    progress_path = os.path.join(args.output_dir, f"progress_{prefix}.json")

    existing_run_totals_df, completed_tasks = load_completed_tasks(run_totals_path, args.resume)
    if completed_tasks:
        print(f"Resuming from {len(completed_tasks)} completed task(s) in {run_totals_path}")

    total_tasks = args.runs * len(GAMES)
    task_idx = len(completed_tasks)
    start_wall = time.time()
    for run_idx in range(1, args.runs + 1):
        for game_idx, game in enumerate(GAMES):
            run_seed = task_seed(args.seed_start, run_idx, game_idx)
            task_id = (run_idx, game)
            if task_id in completed_tasks:
                continue

            task_idx += 1
            random.seed(run_seed)

            t0 = time.time()
            frame = run_one(game)
            elapsed_sec = time.time() - t0

            final_row = frame.iloc[-1]
            total1 = float(final_row["total1"])
            total2 = float(final_row["total2"])
            total_score = total1 + total2

            frame = frame.copy()
            frame.insert(0, "mode", args.mode)
            frame.insert(1, "game", game)
            frame.insert(2, "run", run_idx)
            frame.insert(3, "seed", run_seed)
            frame.insert(4, "final_total_score", total_score)
            append_csv(detail_path, frame)

            run_row_df = pd.DataFrame(
                [
                    {
                        "mode": args.mode,
                        "game": game,
                        "run": run_idx,
                        "seed": run_seed,
                        "rounds": args.rounds,
                        "total1": total1,
                        "total2": total2,
                        "total_score": total_score,
                        "elapsed_sec": elapsed_sec,
                    }
                ]
            )
            append_csv(run_totals_path, run_row_df)
            completed_tasks.add(task_id)

            progress_payload = {
                "mode": args.mode,
                "runs": args.runs,
                "rounds": args.rounds,
                "seed_start": args.seed_start,
                "resume": bool(args.resume),
                "completed_tasks": len(completed_tasks),
                "total_tasks": total_tasks,
                "remaining_tasks": total_tasks - len(completed_tasks),
                "last_completed": {"run": run_idx, "game": game, "seed": run_seed},
                "updated_at_epoch": time.time(),
            }
            write_json_atomic(progress_path, progress_payload)
            print(
                f"[{task_idx}/{total_tasks}] mode={args.mode} game={game} run={run_idx} "
                f"seed={run_seed} total_score={total_score:.2f} elapsed={elapsed_sec:.2f}s"
            )

    if os.path.exists(run_totals_path):
        run_totals_df = pd.read_csv(run_totals_path)
    else:
        run_totals_df = existing_run_totals_df.copy()
    run_totals_df = canonicalize_run_totals(run_totals_df)
    run_totals_df.to_csv(run_totals_path, index=False)

    if os.path.exists(detail_path):
        details_df = pd.read_csv(detail_path)
        details_df = canonicalize_round_logs(details_df)
        details_df.to_csv(detail_path, index=False)
    else:
        details_df = pd.DataFrame()

    summary_df = (
        run_totals_df.groupby(["mode", "game"], as_index=False)
        .agg(
            runs=("run", "count"),
            mean_total_score=("total_score", "mean"),
            std_total_score=("total_score", "std"),
            min_total_score=("total_score", "min"),
            max_total_score=("total_score", "max"),
            mean_total1=("total1", "mean"),
            mean_total2=("total2", "mean"),
            mean_elapsed_sec=("elapsed_sec", "mean"),
        )
        .sort_values(["mode", "game"])
        .reset_index(drop=True)
    )
    summary_df["std_total_score"] = summary_df["std_total_score"].fillna(0.0)

    summary_df.to_csv(summary_path, index=False)

    config_payload = vars(args).copy()
    config_payload["games"] = list(GAMES)
    config_payload["wall_time_sec"] = time.time() - start_wall
    config_payload["completed_tasks"] = int(run_totals_df.shape[0])
    config_payload["total_tasks"] = int(total_tasks)
    write_json_atomic(config_path, config_payload)

    print("\n=== Mean Total Score Summary ===")
    print(summary_df.to_string(index=False))
    print("\nWrote:")
    print(f"  {detail_path}")
    print(f"  {run_totals_path}")
    print(f"  {summary_path}")
    print(f"  {config_path}")

    return {
        "detail_path": detail_path,
        "run_totals_path": run_totals_path,
        "summary_path": summary_path,
        "config_path": config_path,
    }


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
