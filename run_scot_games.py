#!/usr/bin/env python3
"""Run repeated games with SCoT prompting using gpt-oss-20b.

SCoT procedure each round:
1) Predict opponent's next action.
2) Choose own action conditioned on that prediction.
"""

import argparse
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from run_ps_br_games import (
    STRATEGY_DESCRIPTIONS as PSBR_STRATEGY_DESCRIPTIONS,
    collusive_prior_label as psbr_collusive_prior_label,
    game_display_name as psbr_game_display_name,
    strategy_menu as psbr_strategy_menu,
)

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

HARMONY_RULES = (
    "You are playing a game repeatedly with another player. In this game, you can choose "
    "between Action C and Action D. You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose C and the other player chooses C, then you win 3 points and the other player wins 3 points.\n"
    "If you choose C and the other player chooses D, then you win 2 points and the other player wins 1 point.\n"
    "If you choose D and the other player chooses C, then you win 1 point and the other player wins 2 points.\n"
    "If you choose D and the other player chooses D, then you win 0 points and the other player wins 0 points.\n\n"
)

DEADLOCK_RULES = (
    "You are playing a game repeatedly with another player. In this game, you can choose "
    "between Action C and Action D. You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose C and the other player chooses C, then you win 0 points and the other player wins 0 points.\n"
    "If you choose C and the other player chooses D, then you win -1 points and the other player wins 2 points.\n"
    "If you choose D and the other player chooses C, then you win 2 points and the other player wins -1 points.\n"
    "If you choose D and the other player chooses D, then you win 1 point and the other player wins 1 point.\n\n"
)

PROMO_RULES = (
    "You are playing a game repeatedly with another player. In this game, you can choose "
    "between Action R (Regular), Action P (Promotion), and Action Z (Price war). "
    "You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose R and the other player chooses R, then you win 3 points and the other player wins 3 points.\n"
    "If you choose R and the other player chooses P, then you win 1 points and the other player wins 6 points.\n"
    "If you choose R and the other player chooses Z, then you win 0 points and the other player wins 0 points.\n"
    "If you choose P and the other player chooses R, then you win 6 points and the other player wins 1 points.\n"
    "If you choose P and the other player chooses P, then you win 2 points and the other player wins 2 points.\n"
    "If you choose P and the other player chooses Z, then you win 0 points and the other player wins 0 points.\n"
    "If you choose Z and the other player chooses R, then you win 0 points and the other player wins 0 points.\n"
    "If you choose Z and the other player chooses P, then you win 0 points and the other player wins 0 points.\n"
    "If you choose Z and the other player chooses Z, then you win 0 points and the other player wins 0 points.\n\n"
)

COLLUSION_RULES = (
    "You are playing a collusion game repeatedly with another player. "
    "In this game, you can choose between Action K (Cartel / collude), "
    "Action N (Normal competition), and Action P (Punishment / price war). "
    "You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose K and the other player chooses K, then you win 10 points and the other player wins 10 points.\n"
    "If you choose K and the other player chooses N, then you win 2 points and the other player wins 14 points.\n"
    "If you choose K and the other player chooses P, then you win -2 points and the other player wins 3 points.\n"
    "If you choose N and the other player chooses K, then you win 14 points and the other player wins 2 points.\n"
    "If you choose N and the other player chooses N, then you win 4 points and the other player wins 4 points.\n"
    "If you choose N and the other player chooses P, then you win 0 points and the other player wins 0 points.\n"
    "If you choose P and the other player chooses K, then you win 3 points and the other player wins -2 points.\n"
    "If you choose P and the other player chooses N, then you win 0 points and the other player wins 0 points.\n"
    "If you choose P and the other player chooses P, then you win 1 points and the other player wins 1 points.\n\n"
)

SAMARITAN_HELPER_RULES = (
    "You are playing Samaritan's dilemma repeatedly with another player. "
    "You are the Helper and can choose between Option H (Help) and Option N (No-help). "
    "The other player is the Recipient and can choose Option W (Work) or Option S (Shirk). "
    "You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose Option H and the other player chooses Option W, then you win 2 points and the other player wins -1 points.\n"
    "If you choose Option H and the other player chooses Option S, then you win 0 points and the other player wins 0 points.\n"
    "If you choose Option N and the other player chooses Option W, then you win 1 points and the other player wins -2 points.\n"
    "If you choose Option N and the other player chooses Option S, then you win -1 points and the other player wins -3 points.\n\n"
)

SAMARITAN_RECIPIENT_RULES = (
    "You are playing Samaritan's dilemma repeatedly with another player. "
    "You are the Recipient and can choose between Option W (Work) and Option S (Shirk). "
    "The other player is the Helper and can choose Option H (Help) or Option N (No-help). "
    "You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose Option W and the other player chooses Option H, then you win -1 points and the other player wins 2 points.\n"
    "If you choose Option S and the other player chooses Option H, then you win 0 points and the other player wins 0 points.\n"
    "If you choose Option W and the other player chooses Option N, then you win -2 points and the other player wins 1 points.\n"
    "If you choose Option S and the other player chooses Option N, then you win -3 points and the other player wins -1 points.\n\n"
)

LEMONS_SELLER_RULES = (
    "You are playing the Lemons game repeatedly with another player. "
    "You are the Seller and can choose between Option HQ (High-quality) and Option LQ (Low-quality). "
    "The other player is the Buyer and can choose Option B (Buy) or Option D (Don't buy). "
    "You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose Option HQ and the other player chooses Option B, then you win 3 points and the other player wins 3 points.\n"
    "If you choose Option HQ and the other player chooses Option D, then you win -1 points and the other player wins 0 points.\n"
    "If you choose Option LQ and the other player chooses Option B, then you win 4 points and the other player wins -1 points.\n"
    "If you choose Option LQ and the other player chooses Option D, then you win 0 points and the other player wins 0 points.\n\n"
)

LEMONS_BUYER_RULES = (
    "You are playing the Lemons game repeatedly with another player. "
    "You are the Buyer and can choose between Option B (Buy) and Option D (Don't buy). "
    "The other player is the Seller and can choose Option HQ (High-quality) or Option LQ (Low-quality). "
    "You will play {rounds} rounds in total with the same player. "
    "The rules of the game are as follows:\n"
    "If you choose Option B and the other player chooses Option HQ, then you win 3 points and the other player wins 3 points.\n"
    "If you choose Option D and the other player chooses Option HQ, then you win 0 points and the other player wins -1 points.\n"
    "If you choose Option B and the other player chooses Option LQ, then you win -1 points and the other player wins 4 points.\n"
    "If you choose Option D and the other player chooses Option LQ, then you win 0 points and the other player wins 0 points.\n\n"
)

TRAVELERS_RULES_TEMPLATE = (
    "You are playing Traveler's Dilemma repeatedly with another player. "
    "In this game, you must choose one integer claim between {low} and {high} inclusive. "
    "You will play {rounds} rounds in total with the same player.\n"
    "Let m be the lower of the two claims and M be the higher claim.\n"
    "- If both claims are equal (x1 = x2 = m), both players receive m points.\n"
    "- If claims differ, the low claimant receives m + {bonus} points and the high claimant receives m - {bonus} points.\n"
    "Choose one integer claim each round.\n\n"
)

OPTION_TOKEN_PATTERN = re.compile(r"\bOPTION\s*([A-Z0-9]+)\b", flags=re.IGNORECASE)
GENERIC_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|-?\d+")
FIRST_ACTION_CHOICES = ["model", "defect"]


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    backoff_factor: float = 2.0


class OpenAIBackend:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        retry_cfg: RetryConfig,
        max_new_tokens: int,
        parse_retries: int,
    ):
        self.client = client
        self.model_name = model_name
        self.retry_cfg = retry_cfg
        self.max_new_tokens = max_new_tokens
        self.parse_retries = parse_retries

    def choose(self, prompt: str, allowed_actions: List[str]) -> str:
        retries = 0
        attempt_prompt = prompt
        parse_failures = 0
        action_hint = ", ".join(allowed_actions)

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
                    return extract_action(content, allowed_actions)
                except ValueError:
                    parse_failures += 1
                    if parse_failures > self.parse_retries:
                        fallback = deterministic_fallback_action(prompt, allowed_actions)
                        print(
                            "Warning: Could not parse API response after retries; "
                            f"using fallback choice '{fallback}'. Raw response: {content!r}"
                        )
                        return fallback
                    attempt_prompt = (
                        f"{prompt}\n\n"
                        f"IMPORTANT: Respond with exactly one action only from: {action_hint}."
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
        parse_retries: int,
    ):
        if hf_pipeline is None:
            raise RuntimeError(
                "Transformers is not installed. Install with: pip install -U transformers kernels torch"
            )

        import torch

        if torch_dtype == "auto":
            dtype = "auto"
        else:
            dtype = getattr(torch, torch_dtype)

        resolved_device_map = None if device_map == "none" else device_map

        self.max_new_tokens = max_new_tokens
        self.parse_retries = parse_retries

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
        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch
        self.action_id_cache: dict[Tuple[str, ...], dict[int, str]] = {}
        self.pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

    def _render_prompt_for_generation(self, prompt: str) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        # Prefer continuing an empty assistant final message so generation starts
        # in final output mode instead of an analysis channel.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                reasoning_effort="low",
            )
        except Exception:
            # Fallback for chat templates that do not support these kwargs.
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

    def _build_action_id_map(self, allowed_actions: List[str]) -> dict[int, str]:
        cache_key = tuple(sorted({action.strip().upper() for action in allowed_actions if action.strip()}))
        cached = self.action_id_cache.get(cache_key)
        if cached is not None:
            return cached

        id_to_action: dict[int, str] = {}
        for action in cache_key:
            probes = [action, f" {action}", f"\n{action}", f"Option {action}"]
            for probe in probes:
                token_ids = self.tokenizer.encode(probe, add_special_tokens=False)
                if not token_ids:
                    continue
                for token_id in (token_ids[0], token_ids[-1]):
                    decoded = self.tokenizer.decode([token_id]).strip().upper()
                    if decoded == action:
                        id_to_action[token_id] = action
        self.action_id_cache[cache_key] = id_to_action
        return id_to_action

    def _choose_from_constrained_logits(self, prompt: str, allowed_actions: List[str]) -> str:
        torch = self._torch
        model_input_text = self._render_prompt_for_generation(prompt)

        inputs = self.tokenizer(model_input_text, return_tensors="pt")
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :].squeeze(0)

        normalized_actions = {action.strip().upper() for action in allowed_actions if action.strip()}
        id_to_action = self._build_action_id_map(allowed_actions)
        covered_actions = set(id_to_action.values())
        if not normalized_actions.issubset(covered_actions):
            missing = sorted(normalized_actions - covered_actions)
            raise RuntimeError(f"Missing tokenizer IDs for actions: {missing}")

        candidate_ids = sorted(id_to_action.keys())
        candidate_logits = logits[candidate_ids]
        best_idx = int(torch.argmax(candidate_logits).item())
        chosen_token_id = candidate_ids[best_idx]
        return id_to_action[chosen_token_id]

    def choose(self, prompt: str, allowed_actions: List[str]) -> str:
        try:
            return self._choose_from_constrained_logits(prompt, allowed_actions)
        except Exception as exc:
            print(
                "Warning: constrained action selection failed; "
                f"falling back to generation parser. Reason: {exc.__class__.__name__}: {exc}"
            )

        attempt_prompt = prompt
        last_content = ""

        action_hint = ", ".join(allowed_actions)
        for _ in range(self.parse_retries + 1):
            input_payload = self._render_prompt_for_generation(attempt_prompt)
            outputs = self.pipe(
                input_payload,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
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
                if content.startswith(attempt_prompt):
                    content = content[len(attempt_prompt):]
            last_content = content

            try:
                return extract_action(content, allowed_actions)
            except ValueError:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Respond with exactly one action only from: {action_hint}."
                )
                continue

        fallback = deterministic_fallback_action(prompt, allowed_actions)
        print(
            "Warning: Could not parse local HF response after retries; "
            f"using fallback choice '{fallback}'. Last response: {last_content!r}"
        )
        return fallback


class MockBackend:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, prompt: str, allowed_actions: List[str]) -> str:
        del prompt
        if not allowed_actions:
            raise ValueError("allowed_actions must be non-empty")
        return self.rng.choice(allowed_actions)


def deterministic_fallback_action(prompt: str, allowed_actions: List[str]) -> str:
    if not allowed_actions:
        raise ValueError("allowed_actions must be non-empty")
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return allowed_actions[digest[0] % len(allowed_actions)]


def extract_action(raw_text: str, allowed_actions: List[str]) -> str:
    if not allowed_actions:
        raise ValueError("allowed_actions must be non-empty")

    text = raw_text.strip()
    normalized_to_original = {action.strip().upper(): action for action in allowed_actions}

    option_match = OPTION_TOKEN_PATTERN.search(text)
    if option_match:
        candidate = option_match.group(1).strip().upper()
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    for token in GENERIC_TOKEN_PATTERN.findall(text):
        candidate = token.strip().upper()
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    cleaned = text.upper()
    for candidate, original in normalized_to_original.items():
        if cleaned == candidate:
            return original
        if cleaned.startswith(candidate) and (
            len(cleaned) == len(candidate) or not cleaned[len(candidate)].isalnum()
        ):
            return original

    raise ValueError(f"Could not parse model action from: {raw_text!r}")


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


def harmony_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "C" and choice_2 == "C":
        return 3, 3
    if choice_1 == "C" and choice_2 == "D":
        return 2, 1
    if choice_1 == "D" and choice_2 == "C":
        return 1, 2
    if choice_1 == "D" and choice_2 == "D":
        return 0, 0
    raise ValueError(f"Unexpected Harmony actions: {choice_1}, {choice_2}")


def deadlock_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "C" and choice_2 == "C":
        return 0, 0
    if choice_1 == "C" and choice_2 == "D":
        return -1, 2
    if choice_1 == "D" and choice_2 == "C":
        return 2, -1
    if choice_1 == "D" and choice_2 == "D":
        return 1, 1
    raise ValueError(f"Unexpected Deadlock actions: {choice_1}, {choice_2}")


def promo_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "Z" or choice_2 == "Z":
        return 0, 0
    if choice_1 == "R" and choice_2 == "R":
        return 3, 3
    if choice_1 == "R" and choice_2 == "P":
        return 1, 6
    if choice_1 == "P" and choice_2 == "R":
        return 6, 1
    if choice_1 == "P" and choice_2 == "P":
        return 2, 2
    raise ValueError(f"Unexpected Promo actions: {choice_1}, {choice_2}")


def collusion_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "K" and choice_2 == "K":
        return 10, 10
    if choice_1 == "K" and choice_2 == "N":
        return 2, 14
    if choice_1 == "K" and choice_2 == "P":
        return -2, 3
    if choice_1 == "N" and choice_2 == "K":
        return 14, 2
    if choice_1 == "N" and choice_2 == "N":
        return 4, 4
    if choice_1 == "N" and choice_2 == "P":
        return 0, 0
    if choice_1 == "P" and choice_2 == "K":
        return 3, -2
    if choice_1 == "P" and choice_2 == "N":
        return 0, 0
    if choice_1 == "P" and choice_2 == "P":
        return 1, 1
    raise ValueError(f"Unexpected Collusion actions: {choice_1}, {choice_2}")


def samaritan_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "H" and choice_2 == "W":
        return 2, -1
    if choice_1 == "H" and choice_2 == "S":
        return 0, 0
    if choice_1 == "N" and choice_2 == "W":
        return 1, -2
    if choice_1 == "N" and choice_2 == "S":
        return -1, -3
    raise ValueError(f"Unexpected Samaritan actions: {choice_1}, {choice_2}")


def lemons_points(choice_1: str, choice_2: str) -> Tuple[int, int]:
    if choice_1 == "HQ" and choice_2 == "B":
        return 3, 3
    if choice_1 == "HQ" and choice_2 == "D":
        return -1, 0
    if choice_1 == "LQ" and choice_2 == "B":
        return 4, -1
    if choice_1 == "LQ" and choice_2 == "D":
        return 0, 0
    raise ValueError(f"Unexpected Lemons actions: {choice_1}, {choice_2}")


def forced_first_action(game: str, player_idx: int, mode: str) -> Optional[str]:
    if mode == "model":
        return None
    if mode != "defect":
        raise ValueError(f"Unknown first action mode: {mode}")

    if game == "bos":
        if player_idx == 1:
            return "J"
        if player_idx == 2:
            return "F"
        raise ValueError(f"Invalid player index: {player_idx}")
    if game == "pd":
        return "F"
    if game == "harmony":
        return "D"
    if game == "deadlock":
        return "C"
    if game == "promo":
        return "P"
    if game == "collusion":
        return "N"
    if game == "samaritan":
        if player_idx == 1:
            return "H"
        if player_idx == 2:
            return "S"
        raise ValueError(f"Invalid player index: {player_idx}")
    if game == "lemons":
        if player_idx == 1:
            return "LQ"
        if player_idx == 2:
            return "D"
        raise ValueError(f"Invalid player index: {player_idx}")
    raise ValueError(f"Unsupported game for forced first action: {game}")


def travelers_points(claim_1: int, claim_2: int, bonus: int) -> Tuple[int, int]:
    if claim_1 == claim_2:
        return claim_1, claim_2
    if claim_1 < claim_2:
        return claim_1 + bonus, claim_1 - bonus
    return claim_2 - bonus, claim_2 + bonus


def travelers_rules(rounds: int, low: int, high: int, bonus: int) -> str:
    return TRAVELERS_RULES_TEMPLATE.format(rounds=rounds, low=low, high=high, bonus=bonus)


def append_compact_action_history(
    history: str,
    round_idx: int,
    self_action: object,
    opp_action: object,
) -> str:
    if not history:
        history = "Observed action history (your action, opponent action):\n"
    return history + f"Round {round_idx}: {self_action}, {opp_action}\n"


def psbr_prompt_label_description(game: str, label: str, labels: List[str]) -> str:
    if game == "lemons" and label == "grim_forgiving":
        if "always_hq" in labels:
            return (
                "Infer this label if Player A (Seller) starts with HQ, switches to LQ punishment for about 2 "
                "rounds after Buyer plays D, then returns to HQ unless another D occurs."
            )
        if "always_buy" in labels:
            return (
                "Infer this label if Player A (Buyer) starts with B, switches to D punishment for about 2 "
                "rounds after Seller plays LQ, then returns to B unless another LQ occurs."
            )
    return PSBR_STRATEGY_DESCRIPTIONS[game][label]


def build_next_action_strategy_context(
    game: str,
    round_idx: int,
    opponent_idx: int,
    allowed_actions: Optional[List[str]] = None,
) -> str:
    labels = psbr_strategy_menu(game, player_idx=opponent_idx)
    menu_lines = "\n".join(
        [f"- {label}: {psbr_prompt_label_description(game, label, labels)}" for label in labels]
    )
    prior_label = psbr_collusive_prior_label(game, opponent_idx)
    prior_line = ""
    if prior_label in labels:
        prior_line = (
            f"Strongly expect Player A to play with strategy '{prior_label}'.\n"
        )
    allowed_actions_line = ""
    if allowed_actions:
        normalized_actions = [str(action).strip() for action in allowed_actions if str(action).strip()]
        if normalized_actions:
            if all(token.lstrip("-").isdigit() for token in normalized_actions):
                integer_actions = [int(token) for token in normalized_actions]
                low = min(integer_actions)
                high = max(integer_actions)
                if sorted(integer_actions) == list(range(low, high + 1)) and len(integer_actions) > 6:
                    allowed_actions_line = f"Allowed action tokens: integers from {low} to {high} inclusive.\n"
                else:
                    allowed_actions_line = f"Allowed action tokens: {', '.join(normalized_actions)}.\n"
            else:
                allowed_actions_line = f"Allowed action tokens: {', '.join(normalized_actions)}.\n"
    observed_rounds = max(0, round_idx - 1)
    context_end_round = max(0, round_idx - 1)
    return (
        f"In repeated {psbr_game_display_name(game)}, a strategy maps prior history to a player's next action "
        "(possibly probabilistically).\n"
        f"Allowed strategies:\n{menu_lines}\n\n"
        "Role mapping in this prompt:\n"
        "- Player A is the other player.\n"
        "- Player B is you.\n"
        f"Observed rounds so far: {observed_rounds}.\n"
        f"Context: full history prefix up to round {context_end_round}.\n"
        f"{prior_line}"
        f"{allowed_actions_line}"
        "Output rule: do NOT output scores, reasoning, or ranking.\n"
        "Respond with exactly one action only.\n"
    )


def enrich_prompt_with_strategy_context(
    prompt: str,
    game: str,
    round_idx: int,
    opponent_idx: int,
    allowed_actions: Optional[List[str]] = None,
) -> str:
    context = build_next_action_strategy_context(
        game=game,
        round_idx=round_idx,
        opponent_idx=opponent_idx,
        allowed_actions=allowed_actions,
    )
    marker = "\nA:"
    insert_at = prompt.rfind(marker)
    if insert_at == -1:
        return f"{prompt}\n{context}"
    return f"{prompt[:insert_at]}\n{context}{prompt[insert_at:]}"


def run_bos_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question_1 = BOS_PLAYER1_RULES.format(rounds=rounds)
    question_2 = BOS_PLAYER2_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("bos", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("bos", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you predict the other player will choose, Option J or Option F?\n"
            "A: Option"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="bos", round_idx=round_idx, opponent_idx=2, allowed_actions=["J", "F"]
        )
        prediction_1 = backend_1.choose(pred_prompt_1, ["J", "F"])

        action_prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_1} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (Option J and Option F), "
            "compare which gives you a better result, and then choose.\n"
            "Which Option do you think is the best to choose for you in this round, Option J or Option F?\n"
            "Output only one letter: J or F.\n"
            "A: Option"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, ["J", "F"])

        pred_prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you predict the other player will choose, Option J or Option F?\n"
            "A: Option"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="bos", round_idx=round_idx, opponent_idx=1, allowed_actions=["J", "F"]
        )
        prediction_2 = backend_2.choose(pred_prompt_2, ["J", "F"])

        action_prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_2} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (Option J and Option F), "
            "compare which gives you a better result, and then choose.\n"
            "Which Option do you think is the best to choose for you in this round, Option J or Option F?\n"
            "Output only one letter: J or F.\n"
            "A: Option"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, ["J", "F"])

        points_1, points_2 = bos_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_pd_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = PD_RULES.format(rounds=rounds)
    options = ["J", "F"]

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("pd", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("pd", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        order = [0, 1]
        random.shuffle(order)

        opt1 = options[order[0]]
        opt2 = options[order[1]]

        pred_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Which Option do you predict the other player will choose, Option {opt1} or Option {opt2}?\n"
            "A: Option"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="pd", round_idx=round_idx, opponent_idx=2, allowed_actions=["J", "F"]
        )
        prediction_1 = backend_1.choose(pred_prompt_1, ["J", "F"])

        action_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_1} in round {round_idx}, "
            f"imagine the outcome for both of your possible actions (Option {opt1} and Option {opt2}), "
            "compare which gives you a better result, and then choose.\n"
            f"Which Option do you think is the best to choose for you in this round, Option {opt1} or Option {opt2}?\n"
            "Output only one letter: J or F.\n"
            "A: Option"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, ["J", "F"])

        pred_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Which Option do you predict the other player will choose, Option {opt1} or Option {opt2}?\n"
            "A: Option"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="pd", round_idx=round_idx, opponent_idx=1, allowed_actions=["J", "F"]
        )
        prediction_2 = backend_2.choose(pred_prompt_2, ["J", "F"])

        action_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_2} in round {round_idx}, "
            f"imagine the outcome for both of your possible actions (Option {opt1} and Option {opt2}), "
            "compare which gives you a better result, and then choose.\n"
            f"Which Option do you think is the best to choose for you in this round, Option {opt1} or Option {opt2}?\n"
            "Output only one letter: J or F.\n"
            "A: Option"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, ["J", "F"])

        points_1, points_2 = pd_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
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


def run_harmony_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = HARMONY_RULES.format(rounds=rounds)
    allowed_actions = ["C", "D"]

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("harmony", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("harmony", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, C or D?\n"
            "A:"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="harmony", round_idx=round_idx, opponent_idx=2, allowed_actions=allowed_actions
        )
        prediction_1 = backend_1.choose(pred_prompt_1, allowed_actions)

        action_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_1} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (C and D), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, C or D?\n"
            "Output only one action: C or D.\n"
            "A:"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, allowed_actions)

        pred_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, C or D?\n"
            "A:"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="harmony", round_idx=round_idx, opponent_idx=1, allowed_actions=allowed_actions
        )
        prediction_2 = backend_2.choose(pred_prompt_2, allowed_actions)

        action_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_2} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (C and D), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, C or D?\n"
            "Output only one action: C or D.\n"
            "A:"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, allowed_actions)

        points_1, points_2 = harmony_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_deadlock_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = DEADLOCK_RULES.format(rounds=rounds)
    allowed_actions = ["C", "D"]

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("deadlock", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("deadlock", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, C or D?\n"
            "A:"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="deadlock", round_idx=round_idx, opponent_idx=2, allowed_actions=allowed_actions
        )
        prediction_1 = backend_1.choose(pred_prompt_1, allowed_actions)

        action_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_1} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (C and D), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, C or D?\n"
            "Output only one action: C or D.\n"
            "A:"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, allowed_actions)

        pred_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, C or D?\n"
            "A:"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="deadlock", round_idx=round_idx, opponent_idx=1, allowed_actions=allowed_actions
        )
        prediction_2 = backend_2.choose(pred_prompt_2, allowed_actions)

        action_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_2} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (C and D), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, C or D?\n"
            "Output only one action: C or D.\n"
            "A:"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, allowed_actions)

        points_1, points_2 = deadlock_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_promo_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = PROMO_RULES.format(rounds=rounds)
    allowed_actions = ["R", "P", "Z"]

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("promo", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("promo", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, R, P, or Z?\n"
            "A:"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="promo", round_idx=round_idx, opponent_idx=2, allowed_actions=allowed_actions
        )
        prediction_1 = backend_1.choose(pred_prompt_1, allowed_actions)

        action_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_1} in round {round_idx}, "
            "imagine the outcome for your possible actions (R, P, and Z), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, R, P, or Z?\n"
            "Output only one action: R, P, or Z.\n"
            "A:"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, allowed_actions)

        pred_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, R, P, or Z?\n"
            "A:"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="promo", round_idx=round_idx, opponent_idx=1, allowed_actions=allowed_actions
        )
        prediction_2 = backend_2.choose(pred_prompt_2, allowed_actions)

        action_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_2} in round {round_idx}, "
            "imagine the outcome for your possible actions (R, P, and Z), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, R, P, or Z?\n"
            "Output only one action: R, P, or Z.\n"
            "A:"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, allowed_actions)

        points_1, points_2 = promo_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_collusion_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = COLLUSION_RULES.format(rounds=rounds)
    allowed_actions = ["K", "N", "P"]

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("collusion", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("collusion", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, K, N, or P?\n"
            "A:"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="collusion", round_idx=round_idx, opponent_idx=2, allowed_actions=allowed_actions
        )
        prediction_1 = backend_1.choose(pred_prompt_1, allowed_actions)

        action_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_1} in round {round_idx}, "
            "imagine the outcome for your possible actions (K, N, and P), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, K, N, or P?\n"
            "Output only one action: K, N, or P.\n"
            "A:"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, allowed_actions)

        pred_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which action do you predict the other player will choose, K, N, or P?\n"
            "A:"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="collusion", round_idx=round_idx, opponent_idx=1, allowed_actions=allowed_actions
        )
        prediction_2 = backend_2.choose(pred_prompt_2, allowed_actions)

        action_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose {prediction_2} in round {round_idx}, "
            "imagine the outcome for your possible actions (K, N, and P), "
            "compare which gives you a better result, and then choose.\n"
            "Which action do you think is best for you in this round, K, N, or P?\n"
            "Output only one action: K, N, or P.\n"
            "A:"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, allowed_actions)

        points_1, points_2 = collusion_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_samaritan_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question_1 = SAMARITAN_HELPER_RULES.format(rounds=rounds)
    question_2 = SAMARITAN_RECIPIENT_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("samaritan", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("samaritan", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you predict the other player will choose, Option W or Option S?\n"
            "A: Option"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="samaritan", round_idx=round_idx, opponent_idx=2, allowed_actions=["W", "S"]
        )
        prediction_1 = backend_1.choose(pred_prompt_1, ["W", "S"])

        action_prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_1} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (Option H and Option N), "
            "compare which gives you a better result, and then choose.\n"
            "Which Option do you think is best to choose for you in this round, Option H or Option N?\n"
            "Output only one action: H or N.\n"
            "A: Option"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, ["H", "N"])

        pred_prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you predict the other player will choose, Option H or Option N?\n"
            "A: Option"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="samaritan", round_idx=round_idx, opponent_idx=1, allowed_actions=["H", "N"]
        )
        prediction_2 = backend_2.choose(pred_prompt_2, ["H", "N"])

        action_prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_2} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (Option W and Option S), "
            "compare which gives you a better result, and then choose.\n"
            "Which Option do you think is best to choose for you in this round, Option W or Option S?\n"
            "Output only one action: W or S.\n"
            "A: Option"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, ["W", "S"])

        points_1, points_2 = samaritan_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_lemons_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question_1 = LEMONS_SELLER_RULES.format(rounds=rounds)
    question_2 = LEMONS_BUYER_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        forced_1 = forced_first_action("lemons", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("lemons", player_idx=2, mode=first_action_mode) if round_idx == 1 else None

        pred_prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you predict the other player will choose, Option B or Option D?\n"
            "A: Option"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="lemons", round_idx=round_idx, opponent_idx=2, allowed_actions=["B", "D"]
        )
        prediction_1 = backend_1.choose(pred_prompt_1, ["B", "D"])

        action_prompt_1 = (
            f"{question_1}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_1} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (Option HQ and Option LQ), "
            "compare which gives you a better result, and then choose.\n"
            "Which Option do you think is best to choose for you in this round, Option HQ or Option LQ?\n"
            "Output only one action: HQ or LQ.\n"
            "A: Option"
        )
        answer_1 = forced_1 if forced_1 is not None else backend_1.choose(action_prompt_1, ["HQ", "LQ"])

        pred_prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            "Q: Which Option do you predict the other player will choose, Option HQ or Option LQ?\n"
            "A: Option"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="lemons", round_idx=round_idx, opponent_idx=1, allowed_actions=["HQ", "LQ"]
        )
        prediction_2 = backend_2.choose(pred_prompt_2, ["HQ", "LQ"])

        action_prompt_2 = (
            f"{question_2}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose Option {prediction_2} in round {round_idx}, "
            "imagine the outcome for both of your possible actions (Option B and Option D), "
            "compare which gives you a better result, and then choose.\n"
            "Which Option do you think is best to choose for you in this round, Option B or Option D?\n"
            "Output only one action: B or D.\n"
            "A: Option"
        )
        answer_2 = forced_2 if forced_2 is not None else backend_2.choose(action_prompt_2, ["B", "D"])

        points_1, points_2 = lemons_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, answer_1, answer_2)
        history_2 = append_compact_action_history(history_2, round_idx, answer_2, answer_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                prediction_1,
                prediction_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def run_travelers_scot(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    low: int,
    high: int,
    bonus: int,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = travelers_rules(rounds=rounds, low=low, high=high, bonus=bonus)
    allowed_actions = [str(claim) for claim in range(low, high + 1)]

    for round_idx in range(1, rounds + 1):
        pred_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Which integer claim do you predict the other player will choose (between {low} and {high})?\n"
            f"Respond with exactly one integer from {low} to {high}.\n"
            "A:"
        )
        pred_prompt_1 = enrich_prompt_with_strategy_context(
            pred_prompt_1, game="travelers", round_idx=round_idx, opponent_idx=2, allowed_actions=allowed_actions
        )
        prediction_1 = backend_1.choose(pred_prompt_1, allowed_actions)

        action_prompt_1 = (
            f"{question}{history_1}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose claim {prediction_1} in round {round_idx}, "
            f"imagine the outcome for your possible claims between {low} and {high}, "
            "compare which gives you a better result, and then choose.\n"
            f"Which integer claim do you think is best for you in this round (between {low} and {high})?\n"
            f"Output exactly one integer from {low} to {high}.\n"
            "A:"
        )
        answer_1 = backend_1.choose(action_prompt_1, allowed_actions)

        pred_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Which integer claim do you predict the other player will choose (between {low} and {high})?\n"
            f"Respond with exactly one integer from {low} to {high}.\n"
            "A:"
        )
        pred_prompt_2 = enrich_prompt_with_strategy_context(
            pred_prompt_2, game="travelers", round_idx=round_idx, opponent_idx=1, allowed_actions=allowed_actions
        )
        prediction_2 = backend_2.choose(pred_prompt_2, allowed_actions)

        action_prompt_2 = (
            f"{question}{history_2}"
            f"\nYou are currently playing round {round_idx}.\n"
            f"Q: Given that you think the other player will choose claim {prediction_2} in round {round_idx}, "
            f"imagine the outcome for your possible claims between {low} and {high}, "
            "compare which gives you a better result, and then choose.\n"
            f"Which integer claim do you think is best for you in this round (between {low} and {high})?\n"
            f"Output exactly one integer from {low} to {high}.\n"
            "A:"
        )
        answer_2 = backend_2.choose(action_prompt_2, allowed_actions)

        claim_1 = int(answer_1)
        claim_2 = int(answer_2)
        points_1, points_2 = travelers_points(claim_1, claim_2, bonus=bonus)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_compact_action_history(history_1, round_idx, claim_1, claim_2)
        history_2 = append_compact_action_history(history_2, round_idx, claim_2, claim_1)

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                int(prediction_1),
                int(prediction_2),
                claim_1,
                claim_2,
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
            "prediction1",
            "prediction2",
            "answer1",
            "answer2",
            "points1",
            "points2",
            "total1",
            "total2",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated games with SCoT prompting.")
    parser.add_argument(
        "--backend",
        choices=["hf-local", "openai-compatible", "mock"],
        default="hf-local",
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--game",
        choices=["bos", "pd", "deadlock", "promo", "collusion", "samaritan", "lemons", "both", "all"],
        default="both",
        help="Which game to run.",
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name to query.")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per repeated game.")
    parser.add_argument(
        "--bos-output",
        default="scot/experiment_bos_scot_gpt_oss_20b.csv",
        help="Output CSV path for BOS.",
    )
    parser.add_argument(
        "--pd-output",
        default="scot/experiment_pd_scot_gpt_oss_20b.csv",
        help="Output CSV path for PD.",
    )
    parser.add_argument(
        "--deadlock-output",
        default="scot/experiment_deadlock_scot_gpt_oss_20b.csv",
        help="Output CSV path for Deadlock.",
    )
    parser.add_argument(
        "--promo-output",
        default="scot/experiment_promo_scot_gpt_oss_20b.csv",
        help="Output CSV path for Promo.",
    )
    parser.add_argument(
        "--collusion-output",
        default="scot/experiment_collusion_scot_gpt_oss_20b.csv",
        help="Output CSV path for Collusion.",
    )
    parser.add_argument(
        "--samaritan-output",
        default="scot/experiment_samaritan_scot_gpt_oss_20b.csv",
        help="Output CSV path for Samaritan.",
    )
    parser.add_argument(
        "--lemons-output",
        default="scot/experiment_lemons_scot_gpt_oss_20b.csv",
        help="Output CSV path for Lemons.",
    )
    parser.add_argument(
        "--first-action-mode",
        choices=FIRST_ACTION_CHOICES,
        default="model",
        help=(
            "Round-1 action policy. 'model' uses model output. "
            "'defect' uses self-favoring one-shot actions for each game."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

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
        default=4,
        help="Max generated tokens per model call.",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=2,
        help="How many times to re-prompt when model output cannot be parsed.",
    )
    parser.add_argument(
        "--mxfp4-dequantize",
        action="store_true",
        help="Force MXFP4 dequantization to bf16 during model load.",
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

    if args.game == "both":
        selected_games = ["bos", "pd"]
    elif args.game == "all":
        selected_games = ["bos", "pd", "deadlock", "promo", "collusion", "samaritan", "lemons"]
    else:
        selected_games = [args.game]

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
        backend_1 = OpenAIBackend(
            client=client,
            model_name=args.model,
            retry_cfg=retry_cfg,
            max_new_tokens=args.max_new_tokens,
            parse_retries=args.parse_retries,
        )
        backend_2 = backend_1

    elif args.backend == "hf-local":
        backend_1 = HuggingFaceLocalBackend(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            mxfp4_dequantize=args.mxfp4_dequantize,
            experts_implementation=args.experts_implementation,
            parse_retries=args.parse_retries,
        )
        backend_2 = backend_1

    else:
        backend_1 = MockBackend(seed=args.seed)
        backend_2 = backend_1

    if "bos" in selected_games:
        bos_df = run_bos_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.bos_output)
        bos_df.to_csv(args.bos_output, index=False)
        print(f"Wrote BOS SCoT results to {args.bos_output}")

    if "pd" in selected_games:
        pd_df = run_pd_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.pd_output)
        pd_df.to_csv(args.pd_output, index=False)
        print(f"Wrote PD SCoT results to {args.pd_output}")

    if "deadlock" in selected_games:
        deadlock_df = run_deadlock_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.deadlock_output)
        deadlock_df.to_csv(args.deadlock_output, index=False)
        print(f"Wrote Deadlock SCoT results to {args.deadlock_output}")

    if "promo" in selected_games:
        promo_df = run_promo_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.promo_output)
        promo_df.to_csv(args.promo_output, index=False)
        print(f"Wrote Promo SCoT results to {args.promo_output}")

    if "collusion" in selected_games:
        collusion_df = run_collusion_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.collusion_output)
        collusion_df.to_csv(args.collusion_output, index=False)
        print(f"Wrote Collusion SCoT results to {args.collusion_output}")

    if "samaritan" in selected_games:
        samaritan_df = run_samaritan_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.samaritan_output)
        samaritan_df.to_csv(args.samaritan_output, index=False)
        print(f"Wrote Samaritan SCoT results to {args.samaritan_output}")

    if "lemons" in selected_games:
        lemons_df = run_lemons_scot(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            first_action_mode=args.first_action_mode,
        )
        ensure_parent_dir(args.lemons_output)
        lemons_df.to_csv(args.lemons_output, index=False)
        print(f"Wrote Lemons SCoT results to {args.lemons_output}")


if __name__ == "__main__":
    main()
