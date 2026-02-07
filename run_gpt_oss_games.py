#!/usr/bin/env python3
"""Run repeated BOS and PD games with gpt-oss-20b."""

import argparse
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

try:
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
except Exception:  # pragma: no cover - optional dependency at runtime
    APIConnectionError = APIError = RateLimitError = OpenAI = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoModelForCausalLM = None
    AutoTokenizer = None
    Mxfp4Config = None
    hf_pipeline = None


DEFAULT_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("TOGETHER_API_KEY")
)

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")


BOS_PLAYER1_RULES = (
    "You are playing a game repeatedly with another player. In this game, you can choose "
    "between Option J and Option F. You will play {rounds} rounds in total with the same "
    "player. The rules of the game are as follows:\n"
    "If you choose Option J and the other player chooses Option J, then you win 10 points "
    "and the other player wins 7 points.\n"
    "If you choose Option J and the other player chooses Option F, then you win 0 points "
    "and the other player wins 0 points.\n"
    "If you choose Option F and the other player chooses Option J, then you win 0 points "
    "and the other player wins 0 points.\n"
    "If you choose Option F and the other player chooses Option F, then you win 7 points "
    "and the other player wins 10 points.\n\n"
)

BOS_PLAYER2_RULES = (
    "You are playing a game repeatedly with another player. In this game, you can choose "
    "between Option J and Option F. You will play {rounds} rounds in total with the same "
    "player. The rules of the game are as follows:\n"
    "If you choose Option J and the other player chooses Option J, then you win 7 points "
    "and the other player wins 10 points.\n"
    "If you choose Option J and the other player chooses Option F, then you win 0 points "
    "and the other player wins 0 points.\n"
    "If you choose Option F and the other player chooses Option J, then you win 0 points "
    "and the other player wins 0 points.\n"
    "If you choose Option F and the other player chooses Option F, then you win 10 points "
    "and the other player wins 7 points.\n\n"
)

PD_RULES = (
    "You are playing a game repeatedly with another player. In this game, you can choose "
    "between Option J and Option F. You will play {rounds} rounds in total with the same "
    "player. The rules of the game are as follows:\n"
    "If you choose Option J and the other player chooses Option J, then you win 8 points "
    "and the other player wins 8 points.\n"
    "If you choose Option J and the other player chooses Option F, then you win 0 points "
    "and the other player wins 10 points.\n"
    "If you choose Option F and the other player chooses Option J, then you win 10 points "
    "and the other player wins 0 points.\n"
    "If you choose Option F and the other player chooses Option F, then you win 5 points "
    "and the other player wins 5 points.\n\n"
)

CHOICE_PATTERN = re.compile(r"\b([JF])\b", flags=re.IGNORECASE)


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    backoff_factor: float = 2.0


class OpenAIBackend:
    def __init__(self, client: OpenAI, model_name: str, retry_cfg: RetryConfig, max_new_tokens: int):
        self.client = client
        self.model_name = model_name
        self.retry_cfg = retry_cfg
        self.max_new_tokens = max_new_tokens

    def choose(self, prompt: str) -> str:
        retries = 0
        attempt_prompt = prompt
        parse_retries = 2
        parse_failures = 0
        while retries <= self.retry_cfg.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": attempt_prompt}],
                    temperature=0.0,
                    max_tokens=self.max_new_tokens,
                )
                content = response.choices[0].message.content or ""
                try:
                    return extract_choice(content)
                except ValueError:
                    parse_failures += 1
                    if parse_failures > parse_retries:
                        fallback = deterministic_fallback_choice(prompt)
                        print(
                            "Warning: Could not parse API response after retries; "
                            f"using fallback choice '{fallback}'. Raw response: {content!r}"
                        )
                        return fallback
                    attempt_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Respond with exactly one letter only: J or F."
                    )
                    continue
            except (APIError, APIConnectionError, RateLimitError) as exc:
                retries += 1
                if retries > self.retry_cfg.max_retries:
                    raise RuntimeError(
                        f"Failed after {self.retry_cfg.max_retries} retries while querying {self.model_name}"
                    ) from exc
                delay = self.retry_cfg.base_delay * (self.retry_cfg.backoff_factor ** (retries - 1))
                print(f"Request error ({exc.__class__.__name__}); retrying in {delay:.1f}s...")
                time.sleep(delay)

        raise RuntimeError("Unexpected retry loop termination")


class HuggingFaceLocalBackend:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        device_map: str,
        torch_dtype: str,
        mxfp4_dequantize: bool,
        experts_implementation: str,
    ):
        if hf_pipeline is None:
            raise RuntimeError(
                "Transformers is not installed. Install with: pip install -U transformers kernels torch"
            )

        # Import torch lazily so openai-compatible mode does not require it.
        import torch

        if torch_dtype == "auto":
            dtype = "auto"
        else:
            dtype = getattr(torch, torch_dtype)

        resolved_device_map = None if device_map == "none" else device_map

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("Transformers AutoModel/AutoTokenizer are unavailable in this environment.")

        model_kwargs = {
            "dtype": dtype,
            "device_map": resolved_device_map,
            "trust_remote_code": True,
        }
        if experts_implementation != "auto":
            model_kwargs["experts_implementation"] = experts_implementation
        if mxfp4_dequantize:
            if Mxfp4Config is None:
                raise RuntimeError("Transformers MXFP4 components are unavailable in this environment.")
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if experts_implementation != "auto" and hasattr(model, "set_experts_implementation"):
            model.set_experts_implementation(experts_implementation)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

    def choose(self, prompt: str) -> str:
        attempt_prompt = prompt
        parse_retries = 2
        last_content = ""
        for _ in range(parse_retries + 1):
            outputs = self.pipe(
                [{"role": "user", "content": attempt_prompt}],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            generated = outputs[0]["generated_text"]
            if isinstance(generated, list) and generated:
                last_item = generated[-1]
                if isinstance(last_item, dict):
                    content = str(last_item.get("content", ""))
                else:
                    content = str(last_item)
            else:
                content = str(generated)
            last_content = content

            try:
                return extract_choice(content)
            except ValueError:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Respond with exactly one letter only: J or F."
                )
                continue

        fallback = deterministic_fallback_choice(prompt)
        print(
            "Warning: Could not parse local HF response after retries; "
            f"using fallback choice '{fallback}'. Last response: {last_content!r}"
        )
        return fallback


def deterministic_fallback_choice(prompt: str) -> str:
    # Stable fallback avoids aborting long runs when a model emits malformed text.
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return "J" if digest[0] % 2 == 0 else "F"


def extract_choice(raw_text: str) -> str:
    match = CHOICE_PATTERN.search(raw_text.strip())
    if match:
        return match.group(1).upper()

    cleaned = raw_text.strip().upper()
    if cleaned.startswith("J"):
        return "J"
    if cleaned.startswith("F"):
        return "F"

    raise ValueError(f"Could not parse model choice from: {raw_text!r}")


def bos_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "J" and choice_2 == "J":
        return 10, 7
    if choice_1 == "F" and choice_2 == "F":
        return 7, 10
    if choice_1 != choice_2:
        return 0, 0
    raise ValueError(f"Unexpected BOS choices: {choice_1}, {choice_2}")


def pd_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "J" and choice_2 == "J":
        return 8, 8
    if choice_1 == "F" and choice_2 == "F":
        return 5, 5
    if choice_1 == "J" and choice_2 == "F":
        return 0, 10
    if choice_1 == "F" and choice_2 == "J":
        return 10, 0
    raise ValueError(f"Unexpected PD choices: {choice_1}, {choice_2}")


def run_bos(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question_1 = BOS_PLAYER1_RULES.format(rounds=rounds)
    question_2 = BOS_PLAYER2_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you choose, Option J or Option F?\n"
            "A: Option"
        )
        prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you choose, Option J or Option F?\n"
            "A: Option"
        )

        answer_1 = backend_1.choose(prompt_1)
        answer_2 = backend_2.choose(prompt_2)

        points_1, points_2 = bos_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 += (
            f"In round {round_idx}, you chose Option {answer_1} and the other player "
            f"chose Option {answer_2}. Thus, you won {points_1} points and the other "
            f"player won {points_2} points.\n"
        )
        history_2 += (
            f"In round {round_idx}, you chose Option {answer_2} and the other player "
            f"chose Option {answer_1}. Thus, you won {points_2} points and the other "
            f"player won {points_1} points.\n"
        )

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                answer_1,
                answer_2,
                points_1,
                points_2,
                total_1,
                total_2,
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "round",
            "player1",
            "player2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_pd(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = PD_RULES.format(rounds=rounds)
    options = ["J", "F"]

    for round_idx in range(1, rounds + 1):
        order = [0, 1]
        random.shuffle(order)

        prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Which Option do you choose, Option {options[order[0]]} or Option {options[order[1]]}?\n"
            "A: Option"
        )
        prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Which Option do you choose, Option {options[order[0]]} or Option {options[order[1]]}?\n"
            "A: Option"
        )

        answer_1 = backend_1.choose(prompt_1)
        answer_2 = backend_2.choose(prompt_2)

        points_1, points_2 = pd_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 += (
            f"In round {round_idx}, you chose Option {answer_1} and the other player "
            f"chose Option {answer_2}. Thus, you won {points_1} points and the other "
            f"player won {points_2} points.\n"
        )
        history_2 += (
            f"In round {round_idx}, you chose Option {answer_2} and the other player "
            f"chose Option {answer_1}. Thus, you won {points_2} points and the other "
            f"player won {points_1} points.\n"
        )

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                answer_1,
                answer_2,
                points_1,
                points_2,
                total_1,
                total_2,
                order[0],
                order[1],
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "round",
            "player1",
            "player2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
            "option1",
            "option2",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated BOS and/or PD with gpt-oss-20b."
    )
    parser.add_argument(
        "--backend",
        choices=["hf-local", "openai-compatible"],
        default="hf-local",
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--game",
        choices=["bos", "pd", "both"],
        default="both",
        help="Which game to run.",
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name to query.")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per repeated game.")
    parser.add_argument(
        "--bos-output",
        default="bos/experiment_bos_gpt_oss_20b.csv",
        help="Output CSV path for BOS.",
    )
    parser.add_argument(
        "--pd-output",
        default="pd/experiment_pd_gpt_oss_20b.csv",
        help="Output CSV path for PD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used for PD option order randomization).",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL "
            "or OPENROUTER_BASE_URL env var."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help=(
            "API key. Defaults to OPENAI_API_KEY, OPENROUTER_API_KEY, "
            "or TOGETHER_API_KEY env var."
        ),
    )

    parser.add_argument(
        "--device-map",
        default="none",
        help="Transformers device map, e.g. 'none', 'auto', 'cuda', or explicit mapping.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for local HF loading.",
    )

    parser.add_argument("--max-retries", type=int, default=5, help="API retry attempts.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Max generated tokens per move.",
    )
    parser.add_argument(
        "--mxfp4-dequantize",
        action="store_true",
        help="Force MXFP4 dequantization to bf16 during model load (recommended on ROCm if kernels fail).",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=["auto", "batched_mm", "grouped_mm", "eager"],
        default="auto",
        help="MoE expert implementation to request at model load.",
    )

    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.backend == "openai-compatible":
        if OpenAI is None:
            raise SystemExit("openai package is not installed. Install with: pip install openai")

        if not args.api_key and not args.base_url:
            raise SystemExit(
                "Missing API key. Set OPENAI_API_KEY (or pass --api-key). "
                "For local OpenAI-compatible servers, pass --base-url and optionally any dummy key."
            )
        api_key = args.api_key or "DUMMY_KEY"

        client_kwargs = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url

        client = OpenAI(**client_kwargs)
        retry_cfg = RetryConfig(max_retries=args.max_retries)
        backend_1 = OpenAIBackend(client, args.model, retry_cfg, args.max_new_tokens)
        backend_2 = OpenAIBackend(client, args.model, retry_cfg, args.max_new_tokens)

    else:
        backend_1 = HuggingFaceLocalBackend(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            mxfp4_dequantize=args.mxfp4_dequantize,
            experts_implementation=args.experts_implementation,
        )
        backend_2 = backend_1

    if args.game in ("bos", "both"):
        bos_df = run_bos(backend_1, backend_2, args.model, args.rounds)
        ensure_parent_dir(args.bos_output)
        bos_df.to_csv(args.bos_output, index=False)
        print(f"Wrote BOS results to {args.bos_output}")

    if args.game in ("pd", "both"):
        pd_df = run_pd(backend_1, backend_2, args.model, args.rounds)
        ensure_parent_dir(args.pd_output)
        pd_df.to_csv(args.pd_output, index=False)
        print(f"Wrote PD results to {args.pd_output}")


if __name__ == "__main__":
    main()
