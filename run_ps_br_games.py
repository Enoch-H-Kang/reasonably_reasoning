#!/usr/bin/env python3
"""Run repeated games with a PS-BR-style planner.

PS-BR approximation used here (strategy-level):
1. At each round and for each player, infer/sample one opponent strategy from a
   fixed game-specific strategy menu (either by LLM label inference from history
   or by likelihood weighting over the menu).
2. Evaluate candidate self strategies by rolling out the horizon against the
   sampled opponent strategy.
3. Choose the self strategy with the larger sampled continuation value and play
   the current action prescribed by that strategy.
"""

import argparse
import hashlib
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

CHOICE_PATTERN = re.compile(r"\b([JF])\b", flags=re.IGNORECASE)
OPTION_PATTERN = re.compile(r"\bOPTION\s*([JF])\b", flags=re.IGNORECASE)
FIRST_ACTION_CHOICES = ["model", "defect"]

BOS_STRATEGY_MENU = [
    "insist_j",
    "insist_f",
    "wsls_bos",
    "mlur",
    "alternate_phase0",
    "alternate_phase1",
    "noisy_insist_j",
    "noisy_insist_f",
]

PD_STRATEGY_MENU = [
    "allc",
    "alld",
    "soft_allc",
    "soft_alld",
    "tft",
    "wsls",
    "soft_grim_trigger",
    "grim_trigger",
]

HARMONY_DEADLOCK_STRATEGY_MENU = [
    "allc",
    "alld",
    "tft",
    "stft",
    "generous_tft",
    "grim_trigger",
    "wsls_pavlov",
    "random_pc",
]

PROMO_STRATEGY_MENU = [
    "allR",
    "allP",
    "allZ",
    "soft_allR",
    "soft_allP",
    "mad0",
    "mad1",
    "grim_trigger",
]

COLLUSION_STRATEGY_MENU = [
    "allN",
    "allK",
    "soft_K",
    "soft_N",
    "allP",
    "grimP",
    "nashRev",
    "abreuMAD",
]

SAMARITAN_HELPER_STRATEGY_MENU = [
    "always_help",
    "never_help",
    "tft_help",
    "grim_forgive",
    "grim_nohelp",
    "wsls_helper",
    "noisy_help",
    "noisy_nohelp",
]

SAMARITAN_RECIPIENT_STRATEGY_MENU = [
    "always_work",
    "always_shirk",
    "work_if_helped",
    "exploit_help",
    "grim_shirk_after_nohelp",
    "forgiving_work",
    "noisy_work",
    "noisy_shirk",
]

LEMONS_SELLER_STRATEGY_MENU = [
    "always_hq",
    "always_lq",
    "hq_if_bought_last",
    "grim_hq_until_boycott",
    "lq_if_boycott_last",
    "grim_forgiving",
    "noisy_hq",
    "noisy_lq",
]

LEMONS_BUYER_STRATEGY_MENU = [
    "always_buy",
    "never_buy",
    "soft_always_buy",
    "soft_never_buy",
    "tft_buy",
    "generous_buy",
    "grim_boycott",
    "grim_forgiving",
]

TRAVELERS_STRATEGY_MENU = [
    "claim_low",
    "claim_high",
    "claim_mid",
    "match_last_opp",
    "undercut_last_opp",
    "step_down",
    "step_up",
    "random_claim",
]

LABEL_INFER_MAX_NEW_TOKENS = 32

STRATEGY_DESCRIPTIONS = {
    "bos": {
        "insist_j": "Infer this label if Player A chooses J almost every round, with little dependence on history.",
        "insist_f": "Infer this label if Player A chooses F almost every round, with little dependence on history.",
        "wsls_bos": "Infer this label if Player A tends to repeat its previous action after a matched round (A and B chose the same action) and switch after a mismatched round; round 1 can be either J or F.",
        "mlur": "Infer this label if Player A usually repeats its previous action after a matched round, but looks near-random between J and F after mismatches (and in round 1).",
        "alternate_phase0": "Infer this label if Player A follows a stable J, F, J, F, ... alternation starting with J.",
        "alternate_phase1": "Infer this label if Player A follows a stable F, J, F, J, ... alternation starting with F.",
        "noisy_insist_j": "Infer this label if Player A is mostly J with occasional F noise (roughly 90/10).",
        "noisy_insist_f": "Infer this label if Player A is mostly F with occasional J noise (roughly 90/10).",
    },
    "pd": {
        "allc": "Infer this label if Player A cooperates (J) almost every round regardless of history.",
        "alld": "Infer this label if Player A defects (F) almost every round regardless of history.",
        "soft_allc": "Infer this label if Player A is mostly cooperative (J) with occasional defections (F), roughly 90/10.",
        "soft_alld": "Infer this label if Player A is mostly defecting (F) with occasional cooperation (J), roughly 90/10.",
        "tft": "Infer this label if Player A starts with J and then usually copies Player B's last action.",
        "wsls": "Infer this label if Player A tends to repeat its last action after matched rounds and switch after mismatches, typically with a cooperative start (J).",
        "soft_grim_trigger": "Infer this label if Player A is initially cooperative (J), switches to punishment (F) for about 2 rounds after seeing Player B play F, then returns to J unless another F appears.",
        "grim_trigger": "Infer this label if Player A starts cooperative (J) and, after first seeing Player B play F, switches to F permanently.",
    },
    "harmony": {
        "allc": "Infer this label if Player A chooses C almost every round regardless of history.",
        "alld": "Infer this label if Player A chooses D almost every round regardless of history.",
        "tft": "Infer this label if Player A starts with C and then usually copies Player B's last action.",
        "stft": "Infer this label if Player A starts with D and then usually copies Player B's last action.",
        "generous_tft": "Infer this label if Player A mostly behaves like TFT but sometimes forgives after Player B played D (plays C with nontrivial probability, about 0.3).",
        "grim_trigger": "Infer this label if Player A starts with C and, after first seeing Player B play D, switches to D permanently.",
        "wsls_pavlov": "Infer this label if Player A tends to repeat its last action after matched rounds and switch after mismatches, typically starting from C.",
        "random_pc": "Infer this label if Player A appears near-uniform random between C and D with weak history dependence.",
    },
    "deadlock": {
        "allc": "Infer this label if Player A chooses C almost every round regardless of history.",
        "alld": "Infer this label if Player A chooses D almost every round regardless of history.",
        "tft": "Infer this label if Player A starts with C and then usually copies Player B's last action.",
        "stft": "Infer this label if Player A starts with D and then usually copies Player B's last action.",
        "generous_tft": "Infer this label if Player A mostly behaves like TFT but sometimes forgives after Player B played D (plays C with nontrivial probability, about 0.3).",
        "grim_trigger": "Infer this label if Player A starts with C and, after first seeing Player B play D, switches to D permanently.",
        "wsls_pavlov": "Infer this label if Player A tends to repeat its last action after matched rounds and switch after mismatches, typically starting from C.",
        "random_pc": "Infer this label if Player A appears near-uniform random between C and D with weak history dependence.",
    },
    "promo": {
        "allR": "Infer this label if Player A chooses R almost every round regardless of history.",
        "allP": "Infer this label if Player A chooses P almost every round regardless of history.",
        "allZ": "Infer this label if Player A chooses Z almost every round regardless of history.",
        "soft_allR": "Infer this label if Player A is mostly R with occasional P noise (roughly 90/10).",
        "soft_allP": "Infer this label if Player A is mostly P with occasional R noise (roughly 90/10).",
        "mad0": "Infer this label if Player A follows odd-round P / even-round R alternation, but switches to Z punishment for about 2 rounds after an observed deviation from the prescribed path.",
        "mad1": "Infer this label if Player A follows odd-round R / even-round P alternation, but switches to Z punishment for about 2 rounds after an observed deviation from the prescribed path.",
        "grim_trigger": "Infer this label if Player A follows odd-round P / even-round R alternation until the first observed deviation, then switches to Z permanently.",
    },
    "collusion": {
        "allN": "Infer this label if Player A chooses N almost every round regardless of history.",
        "allK": "Infer this label if Player A chooses K almost every round regardless of history.",
        "soft_K": "Infer this label if Player A is mostly K with occasional N noise (roughly 90/10).",
        "soft_N": "Infer this label if Player A is mostly N with occasional K noise (roughly 90/10).",
        "allP": "Infer this label if Player A chooses P almost every round regardless of history.",
        "grimP": "Infer this label if Player A stays at K until first observed deviation from (K,K), then switches to P permanently.",
        "nashRev": "Infer this label if Player A stays at K until first observed deviation from (K,K), then switches to N permanently.",
        "abreuMAD": "Infer this label if Player A plays K on cooperative path, switches to P for about 2 rounds after observed deviation from (K,K), then returns to K.",
    },
    "samaritan": {
        "always_help": "Infer this label if Player A (Helper) chooses H almost every round regardless of history.",
        "never_help": "Infer this label if Player A (Helper) chooses N almost every round regardless of history.",
        "tft_help": "Infer this label if Player A (Helper) starts with H and then chooses H iff Player B (Recipient) played W in the previous round.",
        "grim_forgive": "Infer this label if Player A (Helper) stays at H until seeing S, then punishes with N for about 2 rounds, then returns to H unless S happens again.",
        "grim_nohelp": "Infer this label if Player A (Helper) stays at H until first seeing S, then switches to N permanently.",
        "wsls_helper": "Infer this label if Player A (Helper) tends to repeat its last action after Player B played W, and switch after Player B played S (with H as typical initial action).",
        "noisy_help": "Infer this label if Player A (Helper) is mostly H with occasional N noise (roughly 90/10).",
        "noisy_nohelp": "Infer this label if Player A (Helper) is mostly N with occasional H noise (roughly 90/10).",
        "always_work": "Infer this label if Player A (Recipient) chooses W almost every round regardless of history.",
        "always_shirk": "Infer this label if Player A (Recipient) chooses S almost every round regardless of history.",
        "work_if_helped": "Infer this label if Player A (Recipient) tends to choose W when Player B (Helper) last played H, and S otherwise (round 1 may be random).",
        "exploit_help": "Infer this label if Player A (Recipient) tends to choose W when Player B (Helper) last played N, and S when Player B last played H (round 1 may be random).",
        "grim_shirk_after_nohelp": "Infer this label if Player A (Recipient) plays W until first seeing Player B play N, then switches to S permanently.",
        "forgiving_work": "Infer this label if Player A (Recipient) usually plays W, including frequent forgiveness after Player B played N (W with nontrivial probability, about 0.3).",
        "noisy_work": "Infer this label if Player A (Recipient) is mostly W with occasional S noise (roughly 90/10).",
        "noisy_shirk": "Infer this label if Player A (Recipient) is mostly S with occasional W noise (roughly 90/10).",
    },
    "lemons": {
        "always_hq": "Infer this label if Player A (Seller) chooses HQ almost every round regardless of history.",
        "always_lq": "Infer this label if Player A (Seller) chooses LQ almost every round regardless of history.",
        "hq_if_bought_last": "Infer this label if Player A (Seller) tends to choose HQ when Player B (Buyer) bought last round (B), and LQ when Player B boycotted (D); round 1 may be random.",
        "grim_hq_until_boycott": "Infer this label if Player A (Seller) keeps HQ until first seeing Buyer boycott (D), then switches to LQ permanently.",
        "lq_if_boycott_last": "Infer this label if Player A (Seller) tends to choose LQ when Player B (Buyer) boycotted last round (D), and HQ when Player B bought (B); round 1 may be random.",
        "noisy_hq": "Infer this label if Player A (Seller) is mostly HQ with occasional LQ noise (roughly 90/10).",
        "noisy_lq": "Infer this label if Player A (Seller) is mostly LQ with occasional HQ noise (roughly 90/10).",
        "always_buy": "Infer this label if Player A (Buyer) chooses B almost every round regardless of history.",
        "never_buy": "Infer this label if Player A (Buyer) chooses D almost every round regardless of history.",
        "soft_always_buy": "Infer this label if Player A (Buyer) is mostly buying (B) with occasional boycott (D) noise (roughly 90/10).",
        "soft_never_buy": "Infer this label if Player A (Buyer) is mostly boycotting (D) with occasional buy (B) noise (roughly 90/10).",
        "tft_buy": "Infer this label if Player A (Buyer) tends to buy (B) when Player B (Seller) last played HQ and boycott (D) after LQ; round 1 may be random.",
        "generous_buy": "Infer this label if Player A (Buyer) is mostly cooperative (B), buys if seller last played HQ, and sometimes forgives if seller last played LQ by still buying with nontrivial probability (about 0.3).",
        "grim_boycott": "Infer this label if Player A (Buyer) buys until first seeing Seller choose LQ, then switches to boycott (D) permanently.",
        "grim_forgiving": "Infer this finite-grim label if Player A starts cooperatively (Seller HQ or Buyer B), switches to punishment (Seller LQ or Buyer D) for about 2 rounds after an opponent defection (Seller sees D or Buyer sees LQ), then resets to cooperative behavior.",
    },
    "travelers": {
        "claim_low": "Infer this label if Player A always (or almost always) claims the lower bound L.",
        "claim_high": "Infer this label if Player A always (or almost always) claims the upper bound H.",
        "claim_mid": "Infer this label if Player A consistently claims the midpoint between L and H.",
        "match_last_opp": "Infer this label if Player A starts near H and then usually matches Player B's previous claim.",
        "undercut_last_opp": "Infer this label if Player A starts near H and then usually claims one below Player B's previous claim (clipped at L).",
        "step_down": "Infer this label if Player A starts high and then decreases its claim by 1 each round (down to L).",
        "step_up": "Infer this label if Player A starts low and then increases its claim by 1 each round (up to H).",
        "random_claim": "Infer this label if Player A's claims look near-uniform random over [L, H] with weak dependence on history.",
    },
}


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class PlannerConfig:
    samples: int
    planning_horizon: int
    discount: float
    sample_temperature: float
    strategy_inference: str
    strategy_memory_rounds: int
    collusive_mode: bool


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

    def choose(self, prompt: str, sample: bool = False, temperature: float = 0.0) -> str:
        retries = 0
        attempt_prompt = prompt
        parse_failures = 0

        while retries <= self.retry_cfg.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": attempt_prompt}],
                    temperature=temperature if sample else 0.0,
                    max_tokens=self.max_new_tokens,
                )
                content = response.choices[0].message.content or ""

                try:
                    return extract_choice(content)
                except ValueError:
                    parse_failures += 1
                    if parse_failures > self.parse_retries:
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

    def infer_label(
        self,
        prompt: str,
        labels: List[str],
        sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        retries = 0
        attempt_prompt = prompt
        parse_failures = 0
        allowed = ", ".join(labels)

        while retries <= self.retry_cfg.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": attempt_prompt}],
                    temperature=temperature if sample else 0.0,
                    max_tokens=LABEL_INFER_MAX_NEW_TOKENS,
                )
                content = response.choices[0].message.content or ""

                try:
                    return extract_strategy_label(content, labels)
                except ValueError:
                    parse_failures += 1
                    if parse_failures > self.parse_retries:
                        raise ValueError(
                            "Could not parse strategy label after retries. "
                            f"Last response: {content!r}"
                        )
                    attempt_prompt = (
                        f"{prompt}\n\n"
                        f"IMPORTANT: Respond with exactly one label from: {allowed}"
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

        self.model_name = model_name
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
        self.choice_id_to_label = self._build_choice_id_map(tokenizer)
        if not self.choice_id_to_label:
            raise RuntimeError("Failed to identify tokenizer IDs for choices J/F.")
        self.pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

    def _render_prompt_for_generation(self, prompt: str) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        # For gpt-oss chat templates, continue an empty assistant final message
        # so decoding starts directly in final output instead of an analysis channel.
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
                reasoning_effort="none",
            )
        except Exception:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )

    @staticmethod
    def _build_choice_id_map(tokenizer) -> dict:
        id_to_label = {}
        probes = [
            ("J", "J"),
            ("F", "F"),
            (" J", "J"),
            (" F", "F"),
            ("\nJ", "J"),
            ("\nF", "F"),
            ("Option J", "J"),
            ("Option F", "F"),
        ]
        for text, label in probes:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            # Multi-token probes like "Option J" may start with a shared "Option"
            # token; keep only token IDs that decode to J/F after stripping.
            for token_id in (token_ids[0], token_ids[-1]):
                decoded = tokenizer.decode([token_id]).strip().upper()
                if decoded == "J":
                    id_to_label[token_id] = "J"
                elif decoded == "F":
                    id_to_label[token_id] = "F"
        return id_to_label

    def _choose_from_constrained_logits(self, prompt: str, sample: bool, temperature: float) -> str:
        torch = self._torch
        model_input_text = self._render_prompt_for_generation(prompt)

        inputs = self.tokenizer(model_input_text, return_tensors="pt")
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :].squeeze(0)

        candidate_ids = sorted(self.choice_id_to_label.keys())
        candidate_logits = logits[candidate_ids]

        if sample:
            temp = max(temperature, 1e-5)
            probs = torch.softmax(candidate_logits / temp, dim=0)
            sampled_idx = int(torch.multinomial(probs, num_samples=1).item())
            chosen_token_id = candidate_ids[sampled_idx]
        else:
            best_idx = int(torch.argmax(candidate_logits).item())
            chosen_token_id = candidate_ids[best_idx]

        return self.choice_id_to_label[chosen_token_id]

    def choose(self, prompt: str, sample: bool = False, temperature: float = 0.0) -> str:
        try:
            return self._choose_from_constrained_logits(prompt, sample=sample, temperature=temperature)
        except Exception as exc:
            print(
                "Warning: constrained J/F selection failed; "
                f"falling back to generation parser. Reason: {exc.__class__.__name__}: {exc}"
            )

        attempt_prompt = prompt
        last_content = ""

        for _ in range(self.parse_retries + 1):
            input_payload = self._render_prompt_for_generation(attempt_prompt)

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": sample,
                "return_full_text": False,
            }
            if sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)

            outputs = self.pipe(input_payload, **gen_kwargs)
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

    def infer_label(
        self,
        prompt: str,
        labels: List[str],
        sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        attempt_prompt = prompt
        last_content = ""
        allowed = ", ".join(labels)

        for _ in range(self.parse_retries + 1):
            input_payload = self._render_prompt_for_generation(attempt_prompt)

            gen_kwargs = {
                "max_new_tokens": LABEL_INFER_MAX_NEW_TOKENS,
                "do_sample": sample,
                "return_full_text": False,
            }
            if sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)

            outputs = self.pipe(input_payload, **gen_kwargs)
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
                return extract_strategy_label(content, labels)
            except ValueError:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Respond with exactly one label from: {allowed}"
                )
                continue

        raise ValueError(
            "Could not parse local HF strategy label after retries. "
            f"Last response: {last_content!r}"
        )

class MockBackend:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, prompt: str, sample: bool = False, temperature: float = 0.0) -> str:  # noqa: ARG002
        if sample:
            return "J" if self.rng.random() < 0.5 else "F"
        return deterministic_fallback_choice(prompt)

    def infer_label(
        self,
        prompt: str,
        labels: List[str],
        sample: bool = False,
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        if sample:
            return self.rng.choice(labels)
        return deterministic_choice_from_options(f"{prompt}|labels|{','.join(labels)}", labels)

def deterministic_fallback_choice(prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return "J" if digest[0] % 2 == 0 else "F"


def extract_choice(raw_text: str) -> str:
    text = raw_text.strip()

    match = OPTION_PATTERN.search(text)
    if match:
        return match.group(1).upper()

    match = CHOICE_PATTERN.search(text)
    if match:
        return match.group(1).upper()

    cleaned = text.upper()
    if cleaned.startswith("J"):
        return "J"
    if cleaned.startswith("F"):
        return "F"

    raise ValueError(f"Could not parse model choice from: {raw_text!r}")


def _label_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))


def extract_strategy_label(raw_text: str, labels: List[str]) -> str:
    if not labels:
        raise ValueError("labels must be non-empty")

    tokens = _label_tokens(raw_text)
    matches: List[str] = []
    for label in labels:
        lt = _label_tokens(label)
        if not lt:
            continue
        n = len(lt)
        for start in range(len(tokens) - n + 1):
            if tokens[start : start + n] == lt:
                matches.append(label)
                break

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]

    cleaned = raw_text.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    if cleaned in labels:
        return cleaned

    first_line = raw_text.strip().splitlines()[0] if raw_text.strip() else ""
    first_line_clean = first_line.lower().replace("-", "_").replace(" ", "_")
    first_line_clean = re.sub(r"[^a-z0-9_]", "", first_line_clean)
    if first_line_clean in labels:
        return first_line_clean

    raise ValueError(f"Could not parse strategy label from: {raw_text!r}")


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


def travelers_points(claim_1: int, claim_2: int, bonus: int) -> Tuple[int, int]:
    if claim_1 == claim_2:
        return claim_1, claim_2
    if claim_1 < claim_2:
        return claim_1 + bonus, claim_1 - bonus
    return claim_2 - bonus, claim_2 + bonus


def travelers_rules(rounds: int, low: int, high: int, bonus: int) -> str:
    return TRAVELERS_RULES_TEMPLATE.format(rounds=rounds, low=low, high=high, bonus=bonus)


def game_display_name(game: str) -> str:
    names = {
        "bos": "Battle of the Sexes",
        "pd": "Prisoner's Dilemma",
        "harmony": "Harmony",
        "deadlock": "Deadlock",
        "promo": "Promo",
        "collusion": "Collusion",
        "samaritan": "Samaritan's Dilemma",
        "lemons": "Lemons",
        "travelers": "Traveler's Dilemma",
    }
    if game not in names:
        raise ValueError(f"Unknown game: {game}")
    return names[game]


def binary_action_labels(game: str, player_idx: Optional[int] = None) -> Tuple[str, str]:
    if game in ("bos", "pd"):
        return "J", "F"
    if game in ("harmony", "deadlock"):
        return "C", "D"
    if game == "samaritan":
        if player_idx == 1:
            return "H", "N"
        if player_idx == 2:
            return "W", "S"
        raise ValueError("player_idx must be 1 or 2 for samaritan")
    if game == "lemons":
        if player_idx == 1:
            return "HQ", "LQ"
        if player_idx == 2:
            return "B", "D"
        raise ValueError("player_idx must be 1 or 2 for lemons")
    raise ValueError(f"Game does not have binary labels: {game}")


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


def stage_points(game: str, choice_1: str, choice_2: str) -> Tuple[int, int]:
    if game == "bos":
        return bos_points(choice_1, choice_2)
    if game == "pd":
        return pd_points(choice_1, choice_2)
    if game == "harmony":
        return harmony_points(choice_1, choice_2)
    if game == "deadlock":
        return deadlock_points(choice_1, choice_2)
    if game == "promo":
        return promo_points(choice_1, choice_2)
    if game == "collusion":
        return collusion_points(choice_1, choice_2)
    if game == "samaritan":
        return samaritan_points(choice_1, choice_2)
    if game == "lemons":
        return lemons_points(choice_1, choice_2)
    raise ValueError(f"Unknown game: {game}")


def player_payoff(game: str, player_idx: int, self_action: str, opp_action: str) -> Tuple[int, int]:
    if player_idx == 1:
        p_self, p_opp = stage_points(game, self_action, opp_action)
    elif player_idx == 2:
        p_opp, p_self = stage_points(game, opp_action, self_action)
    else:
        raise ValueError(f"Invalid player index: {player_idx}")
    return p_self, p_opp


def travelers_player_payoff(player_idx: int, self_claim: int, opp_claim: int, bonus: int) -> Tuple[int, int]:
    if player_idx == 1:
        p_self, p_opp = travelers_points(self_claim, opp_claim, bonus)
    elif player_idx == 2:
        p_opp, p_self = travelers_points(opp_claim, self_claim, bonus)
    else:
        raise ValueError(f"Invalid player index: {player_idx}")
    return p_self, p_opp


def append_history(
    history: str,
    round_idx: int,
    self_action: str,
    opp_action: str,
    self_points: int,
    opp_points: int,
) -> str:
    return (
        history
        + f"In round {round_idx}, you chose Option {self_action} and the other player "
        + f"chose Option {opp_action}. Thus, you won {self_points} points and the other "
        + f"player won {opp_points} points.\n"
    )


def append_travelers_history(
    history: str,
    round_idx: int,
    self_claim: int,
    opp_claim: int,
    self_points: int,
    opp_points: int,
) -> str:
    return (
        history
        + f"In round {round_idx}, you chose claim {self_claim} and the other player "
        + f"chose claim {opp_claim}. Thus, you won {self_points} points and the other "
        + f"player won {opp_points} points.\n"
    )


def deterministic_choice_from_options(key: str, options: List[str]) -> str:
    if not options:
        raise ValueError("options must be non-empty")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return sorted(options)[digest[0] % len(options)]


def strategy_menu(game: str, player_idx: Optional[int] = None) -> List[str]:
    if game == "bos":
        return BOS_STRATEGY_MENU
    if game == "pd":
        return PD_STRATEGY_MENU
    if game in ("harmony", "deadlock"):
        return HARMONY_DEADLOCK_STRATEGY_MENU
    if game == "promo":
        return PROMO_STRATEGY_MENU
    if game == "collusion":
        return COLLUSION_STRATEGY_MENU
    if game == "samaritan":
        if player_idx == 1:
            return SAMARITAN_HELPER_STRATEGY_MENU
        if player_idx == 2:
            return SAMARITAN_RECIPIENT_STRATEGY_MENU
        raise ValueError("player_idx must be 1 or 2 for samaritan")
    if game == "lemons":
        if player_idx == 1:
            return LEMONS_SELLER_STRATEGY_MENU
        if player_idx == 2:
            return LEMONS_BUYER_STRATEGY_MENU
        raise ValueError("player_idx must be 1 or 2 for lemons")
    if game == "travelers":
        return TRAVELERS_STRATEGY_MENU
    raise ValueError(f"Unknown game: {game}")


def clamp_claim(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def travelers_deterministic_claim(
    strategy: str,
    self_history_claims: List[int],
    opp_history_claims: List[int],
    round_idx: int,
    low: int,
    high: int,
) -> int:
    if strategy == "claim_low":
        return low
    if strategy == "claim_high":
        return high
    if strategy == "claim_mid":
        return (low + high) // 2
    if strategy == "match_last_opp":
        if not opp_history_claims:
            return high
        return clamp_claim(opp_history_claims[-1], low, high)
    if strategy == "undercut_last_opp":
        if not opp_history_claims:
            return high
        return clamp_claim(opp_history_claims[-1] - 1, low, high)
    if strategy == "step_down":
        return clamp_claim(high - (round_idx - 1), low, high)
    if strategy == "step_up":
        return clamp_claim(low + (round_idx - 1), low, high)
    if strategy == "random_claim":
        return (low + high) // 2
    raise ValueError(f"Unknown Traveler's Dilemma strategy: {strategy}")


def travelers_claim_from_strategy(
    strategy: str,
    self_history_claims: List[int],
    opp_history_claims: List[int],
    round_idx: int,
    low: int,
    high: int,
) -> int:
    if strategy == "random_claim":
        return random.randint(low, high)
    return travelers_deterministic_claim(strategy, self_history_claims, opp_history_claims, round_idx, low, high)


def collusive_prior_label(game: str, opponent_idx: int) -> Optional[str]:
    if game == "bos":
        return "alternate_phase0"
    if game == "pd":
        return "soft_grim_trigger"
    if game == "promo":
        if opponent_idx == 1:
            return "mad0"
        if opponent_idx == 2:
            return "mad1"
        raise ValueError(f"Invalid player index: {opponent_idx}")
    if game == "collusion":
        if opponent_idx in (1, 2):
            return "abreuMAD"
        raise ValueError(f"Invalid player index: {opponent_idx}")
    if game == "samaritan":
        if opponent_idx == 1:
            return "grim_forgive"
        if opponent_idx == 2:
            return "noisy_work"
        raise ValueError(f"Invalid player index: {opponent_idx}")
    if game == "lemons":
        if opponent_idx in (1, 2):
            return "grim_forgiving"
        raise ValueError(f"Invalid player index: {opponent_idx}")
    return None


def build_strategy_inference_prompt(
    game: str,
    round_idx: int,
    rules_text: str,
    history_opp_self: List[Tuple[object, object]],
    labels: List[str],
    history_start_round: int = 1,
    collusive_prior: Optional[str] = None,
) -> str:
    def prompt_label_description(label: str) -> str:
        # Lemons reuses the same label name for both roles; disambiguate text by menu role.
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
        return STRATEGY_DESCRIPTIONS[game][label]

    game_name = game_display_name(game)
    menu_lines = "\n".join([f"- {label}: {prompt_label_description(label)}" for label in labels])
    observed_rounds = max(0, round_idx - 1)
    target_round = max(0, round_idx - 1)
    if history_opp_self:
        target_round = history_start_round + len(history_opp_self) - 1
    context_end_round = max(0, target_round - 1)
    if history_start_round <= 1:
        context_line = f"Context: full history prefix up to round {context_end_round}."
    else:
        context_line = (
            f"Context: recent history window from round {history_start_round} to round {context_end_round}."
        )
    action_history_lines = [
        "Observed action history tuple format: (Player A action, Player B action).",
        "Player A is the opponent whose strategy label you must infer.",
        "Player B is you (the decision-maker).",
    ]
    action_history_lines.extend(
        [
        # "Infer the opponent strategy in two steps:",
        # "1) Internally infer opponent's behavior type from recent history:",
        # "- history-independent (mostly the same action regardless of your past actions),",
        # "- reactive (depends on your previous actions),",
        # "- periodic (alternating/cycle),",
        # "- or a noisy variant of one of the above.",
        # "2) Map that inferred behavior to exactly one allowed label.",
        # "Internally infer opponent's behavior pattern from recent history, and map that inferred behavior to exactly one allowed label.",
        # "Primary goal: maximize one-step-ahead prediction of the opponent's next action.",
        # "Evaluate fit over all observed rounds, with extra weight on recent rounds.",
        # "Prefer the simplest label that fits the data; do not infer complex periodic/reactive",
        # "structure unless the history clearly supports it across multiple rounds.",
        # "If recent rounds conflict with earlier rounds, treat this as a possible strategy shift and",
        # "prioritize the most recent consistent behavior.",
        # "If multiple labels are close, break ties by: (1) better recent predictive fit, then",
        # "(2) simpler label.",
        # "For each t from 2 to T+1 (where T is the number of observed rounds), use history",
        # "from rounds 1..t-2 as context, and opponent action at round t-1 as the target.",
        context_line,
        ]
    )
    if history_opp_self:
        action_history_lines.extend(
            [
                f"Target: observed Player A action at round {target_round}.",
                "Choose the allowed label that makes this observed Player A target most compatible with the context.",
            ]
        )
    else:
        action_history_lines.extend(
            [
                "No observed Player A actions yet in this repeated game.",
                "Choose the allowed Player A's strategy label that best reflects the history.",
            ]
        )
    # for idx, (opp_action, self_action) in enumerate(history_opp_self, start=1):
    #     action_history_lines.append(f"Logged round {idx}: opp={opp_action}, you={self_action}")
    #     prefix_desc = "empty history (no prior rounds)" if idx == 1 else f"logged rounds 1..{idx - 1}"
    #     action_history_lines.append(
    #         f"Prefix-target pair {idx}: context={prefix_desc}; target opp action at logged round {idx}={opp_action}"
    #     )
    if history_opp_self:
        idx = len(history_opp_self)
        opp_action = history_opp_self[-1][0]
        # action_history_lines.append(f"Logged round {idx}: opp={opp_action}, you={self_action}")
        # prefix_desc = "empty history (no prior rounds)" if idx == 1 else f"logged rounds 1..{idx - 1}"
        # action_history_lines.append(
        #     f"Prefix-target pair {idx}: context={prefix_desc}; target opp action at logged round {idx}={opp_action}"
        # )
        # if idx == 1:
        #     context_history_text = "empty history"
        # else:
        #     context_history_text = "; ".join(
        #         [
        #             f"round {hist_idx}: opp={hist_opp}, you={hist_you}"
        #             for hist_idx, (hist_opp, hist_you) in enumerate(history_opp_self[:-1], start=1)
        #         ]
        #     )
        # action_history_lines.append(f"At round {idx + 1}:")
        # action_history_lines.append(f'History up to round {idx - 1} was: "{context_history_text}"')
        # action_history_lines.append(
        #     f"The opponent's action at round {idx}, responding to the history up to round {idx - 1}, was {opp_action}."
        # )
        target_round = history_start_round + idx - 1
        prefix_history = history_opp_self[:-1]
        action_history_lines.append(f"At round {target_round + 1}, use this mapping:")
        if prefix_history:
            action_history_lines.append(
                f"Context history as (Player A, Player B), rounds {history_start_round}..{target_round - 1}:"
            )
            for hist_idx, (hist_opp, hist_you) in enumerate(prefix_history, start=history_start_round):
                action_history_lines.append(f"round {hist_idx}: Player A={hist_opp}, Player B={hist_you}")
        else:
            action_history_lines.append(
                f"Context history rounds {history_start_round}..{target_round - 1} as (Player A, Player B): empty"
            )
        action_history_lines.append(f"Observed target Player A action at round {target_round}: {opp_action}")
    action_history_text = "\n".join(action_history_lines) + "\n\n"
    tail_guidance_lines: List[str] = []
    if collusive_prior is not None:
        tail_guidance_lines.append(f"Strongly expect Player A to play with strategy '{collusive_prior}'.")
    if history_opp_self:
        tail_guidance_lines.append(
            "Player A's strategy may have changed over time, so weigh recent rounds more heavily than earlier rounds."
        )
    tail_guidance_text = "".join([f"{line}\n" for line in tail_guidance_lines])
    return (
        f"You are inferring Player A's strategy (the opponent) in repeated {game_name}.\n"
        "In this repeated-game setting, a strategy is a rule that maps prior history to the player's next action (possibly probabilistically).\n"
        f"{rules_text}"
        f"Observed rounds so far: {observed_rounds}.\n"
        "\n"
        f"Allowed labels:\n{menu_lines}\n\n"
        f"{action_history_text}"
        f"{tail_guidance_text}"
        "Output rule: do NOT output scores, reasoning, or ranking.\n"
        "Respond with exactly one label only.\n\n"
        "**Output only the label.**\n"
    )


PROMO_ACTION_ORDER = ("R", "P", "Z")
COLLUSION_ACTION_ORDER = ("K", "N", "P")


def promo_phase_action(phase: int, round_idx: int) -> str:
    if phase not in (0, 1):
        raise ValueError(f"Invalid promo phase: {phase}")
    if round_idx <= 0:
        raise ValueError(f"round_idx must be >= 1, got {round_idx}")
    is_odd_round = round_idx % 2 == 1
    if phase == 0:
        return "P" if is_odd_round else "R"
    return "R" if is_odd_round else "P"


def promo_phase_expected_pair(phase: int, round_idx: int) -> Tuple[str, str]:
    self_action = promo_phase_action(phase, round_idx)
    opp_action = "R" if self_action == "P" else "P"
    return self_action, opp_action


def promo_mad_punishment_timer(
    phase: int,
    self_history_actions: List[str],
    opp_history_actions: List[str],
) -> int:
    if len(self_history_actions) != len(opp_history_actions):
        raise ValueError("self_history_actions and opp_history_actions must have equal length")

    punish_remaining = 0
    for past_round, (self_action, opp_action) in enumerate(
        zip(self_history_actions, opp_history_actions), start=1
    ):
        if punish_remaining > 0:
            expected_self, expected_opp = "Z", "Z"
        else:
            expected_self, expected_opp = promo_phase_expected_pair(phase, past_round)
        deviated = (self_action != expected_self) or (opp_action != expected_opp)
        if deviated:
            punish_remaining = 2
        elif punish_remaining > 0:
            punish_remaining -= 1
    return punish_remaining


def promo_grim_triggered(
    phase: int,
    self_history_actions: List[str],
    opp_history_actions: List[str],
) -> bool:
    if len(self_history_actions) != len(opp_history_actions):
        raise ValueError("self_history_actions and opp_history_actions must have equal length")

    triggered = False
    for past_round, (self_action, opp_action) in enumerate(
        zip(self_history_actions, opp_history_actions), start=1
    ):
        if triggered:
            expected_self, expected_opp = "Z", "Z"
        else:
            expected_self, expected_opp = promo_phase_expected_pair(phase, past_round)
        deviated = (self_action != expected_self) or (opp_action != expected_opp)
        if deviated:
            triggered = True
    return triggered


def promo_action_probabilities(
    strategy: str,
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> dict[str, float]:
    if strategy == "allR":
        return {"R": 1.0}
    if strategy == "allP":
        return {"P": 1.0}
    if strategy == "allZ":
        return {"Z": 1.0}
    if strategy == "soft_allR":
        return {"R": 0.9, "P": 0.1}
    if strategy == "soft_allP":
        return {"P": 0.9, "R": 0.1}
    if strategy == "mad0":
        punish_remaining = promo_mad_punishment_timer(0, self_history_actions, opp_history_actions)
        if punish_remaining > 0:
            return {"Z": 1.0}
        return {promo_phase_action(0, round_idx): 1.0}
    if strategy == "mad1":
        punish_remaining = promo_mad_punishment_timer(1, self_history_actions, opp_history_actions)
        if punish_remaining > 0:
            return {"Z": 1.0}
        return {promo_phase_action(1, round_idx): 1.0}
    if strategy == "grim_trigger":
        if promo_grim_triggered(0, self_history_actions, opp_history_actions):
            return {"Z": 1.0}
        return {promo_phase_action(0, round_idx): 1.0}
    raise ValueError(f"Unknown Promo strategy: {strategy}")


def promo_strategy_mode_action(
    strategy: str,
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> str:
    probs = promo_action_probabilities(strategy, self_history_actions, opp_history_actions, round_idx)
    best_prob = max(probs.values())
    for action in PROMO_ACTION_ORDER:
        if abs(probs.get(action, 0.0) - best_prob) < 1e-12:
            return action
    raise RuntimeError(f"Could not resolve mode action for Promo strategy: {strategy}")


def collusion_deviation_triggered(
    self_history_actions: List[str],
    opp_history_actions: List[str],
) -> bool:
    if len(self_history_actions) != len(opp_history_actions):
        raise ValueError("self_history_actions and opp_history_actions must have equal length")

    for self_action, opp_action in zip(self_history_actions, opp_history_actions):
        if self_action != "K" or opp_action != "K":
            return True
    return False


def collusion_abreu_punishment_timer(
    self_history_actions: List[str],
    opp_history_actions: List[str],
) -> int:
    if len(self_history_actions) != len(opp_history_actions):
        raise ValueError("self_history_actions and opp_history_actions must have equal length")

    punish_remaining = 0
    for self_action, opp_action in zip(self_history_actions, opp_history_actions):
        if punish_remaining > 0:
            punish_remaining -= 1
            continue
        if self_action != "K" or opp_action != "K":
            punish_remaining = 2
    return punish_remaining


def collusion_action_probabilities(
    strategy: str,
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> dict[str, float]:
    if strategy == "allN":
        return {"N": 1.0}
    if strategy == "allK":
        return {"K": 1.0}
    if strategy == "soft_K":
        return {"K": 0.9, "N": 0.1}
    if strategy == "soft_N":
        return {"N": 0.9, "K": 0.1}
    if strategy == "allP":
        return {"P": 1.0}
    if strategy == "grimP":
        if collusion_deviation_triggered(self_history_actions, opp_history_actions):
            return {"P": 1.0}
        return {"K": 1.0}
    if strategy == "nashRev":
        if collusion_deviation_triggered(self_history_actions, opp_history_actions):
            return {"N": 1.0}
        return {"K": 1.0}
    if strategy == "abreuMAD":
        if collusion_abreu_punishment_timer(self_history_actions, opp_history_actions) > 0:
            return {"P": 1.0}
        return {"K": 1.0}
    raise ValueError(f"Unknown Collusion strategy: {strategy}")


def collusion_strategy_mode_action(
    strategy: str,
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> str:
    probs = collusion_action_probabilities(strategy, self_history_actions, opp_history_actions, round_idx)
    best_prob = max(probs.values())
    for action in COLLUSION_ACTION_ORDER:
        if abs(probs.get(action, 0.0) - best_prob) < 1e-12:
            return action
    raise RuntimeError(f"Could not resolve mode action for Collusion strategy: {strategy}")


def strategy_prob_j(
    game: str,
    player_idx: int,
    strategy: str,
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> float:
    if game == "bos":
        if strategy == "insist_j":
            return 1.0
        if strategy == "insist_f":
            return 0.0
        if strategy == "wsls_bos":
            if not self_history_actions:
                return 0.5
            last_self = self_history_actions[-1]
            last_opp = opp_history_actions[-1]
            if last_self == last_opp:
                next_action = last_self
            else:
                next_action = "F" if last_self == "J" else "J"
            return 1.0 if next_action == "J" else 0.0
        if strategy == "mlur":
            if not self_history_actions:
                return 0.5
            last_self = self_history_actions[-1]
            last_opp = opp_history_actions[-1]
            # MLUR: lock action on match; randomize after mismatch.
            if last_self != last_opp:
                return 0.5
            next_action = last_self
            return 1.0 if next_action == "J" else 0.0
        if strategy == "alternate_phase0":
            return 1.0 if round_idx % 2 == 1 else 0.0
        if strategy == "alternate_phase1":
            return 0.0 if round_idx % 2 == 1 else 1.0
        if strategy == "noisy_insist_j":
            return 0.9
        if strategy == "noisy_insist_f":
            return 0.1
        raise ValueError(f"Unknown BOS strategy: {strategy}")

    if game in ("pd", "harmony", "deadlock"):
        coop_label, defect_label = binary_action_labels(game, player_idx=player_idx)
        if strategy == "allc":
            return 1.0
        if strategy == "alld":
            return 0.0
        if strategy == "soft_allc":
            return 0.9
        if strategy == "soft_alld":
            return 0.1
        if strategy == "tft":
            if not opp_history_actions:
                return 1.0
            return 1.0 if opp_history_actions[-1] == coop_label else 0.0
        if strategy == "stft":
            if not opp_history_actions:
                return 0.0
            return 1.0 if opp_history_actions[-1] == coop_label else 0.0
        if strategy == "generous_tft":
            if not opp_history_actions:
                return 1.0
            if opp_history_actions[-1] == coop_label:
                return 1.0
            return 0.3
        if strategy == "soft_grim_trigger":
            # Finite-punishment grim trigger: punish for 2 rounds after any recent opponent defection.
            return 0.0 if defect_label in opp_history_actions[-2:] else 1.0
        if strategy == "grim_trigger":
            return 0.0 if defect_label in opp_history_actions else 1.0
        if strategy == "wsls":
            if not self_history_actions:
                return 1.0
            last_self = self_history_actions[-1]
            last_opp = opp_history_actions[-1]
            if last_self == last_opp:
                return 1.0 if last_self == coop_label else 0.0
            switched = defect_label if last_self == coop_label else coop_label
            return 1.0 if switched == coop_label else 0.0
        if strategy == "wsls_pavlov":
            if not self_history_actions:
                return 1.0
            last_self = self_history_actions[-1]
            last_opp = opp_history_actions[-1]
            if last_self == last_opp:
                return 1.0 if last_self == coop_label else 0.0
            switched = defect_label if last_self == coop_label else coop_label
            return 1.0 if switched == coop_label else 0.0
        if strategy == "random_pc":
            return 0.5
        raise ValueError(f"Unknown strategy for {game}: {strategy}")

    if game == "samaritan":
        if player_idx == 1:
            if strategy == "always_help":
                return 1.0
            if strategy == "never_help":
                return 0.0
            if strategy == "tft_help":
                if not opp_history_actions:
                    return 1.0
                return 1.0 if opp_history_actions[-1] == "W" else 0.0
            if strategy == "grim_forgive":
                # Finite grim: punish with N for 2 rounds after any recent recipient shirk (S).
                return 0.0 if "S" in opp_history_actions[-2:] else 1.0
            if strategy == "grim_nohelp":
                return 0.0 if "S" in opp_history_actions else 1.0
            if strategy == "wsls_helper":
                if not self_history_actions:
                    return 1.0
                last_self = self_history_actions[-1]
                last_opp = opp_history_actions[-1]
                if last_opp == "W":
                    return 1.0 if last_self == "H" else 0.0
                switched = "N" if last_self == "H" else "H"
                return 1.0 if switched == "H" else 0.0
            if strategy == "noisy_help":
                return 0.9
            if strategy == "noisy_nohelp":
                return 0.1
            raise ValueError(f"Unknown helper strategy for {game}: {strategy}")

        if player_idx == 2:
            if strategy == "always_work":
                return 1.0
            if strategy == "always_shirk":
                return 0.0
            if strategy == "work_if_helped":
                if not opp_history_actions:
                    return 0.5
                return 1.0 if opp_history_actions[-1] == "H" else 0.0
            if strategy == "exploit_help":
                if not opp_history_actions:
                    return 0.5
                return 1.0 if opp_history_actions[-1] == "N" else 0.0
            if strategy == "grim_shirk_after_nohelp":
                return 0.0 if "N" in opp_history_actions else 1.0
            if strategy == "forgiving_work":
                if not opp_history_actions:
                    return 1.0
                if opp_history_actions[-1] == "H":
                    return 1.0
                return 0.3
            if strategy == "noisy_work":
                return 0.9
            if strategy == "noisy_shirk":
                return 0.1
            raise ValueError(f"Unknown recipient strategy for {game}: {strategy}")

        raise ValueError(f"Invalid player index: {player_idx}")

    if game == "lemons":
        if player_idx == 1:
            if strategy == "always_hq":
                return 1.0
            if strategy == "always_lq":
                return 0.0
            if strategy == "hq_if_bought_last":
                if not opp_history_actions:
                    return 0.5
                return 1.0 if opp_history_actions[-1] == "B" else 0.0
            if strategy == "grim_hq_until_boycott":
                return 0.0 if "D" in opp_history_actions else 1.0
            if strategy == "lq_if_boycott_last":
                if not opp_history_actions:
                    return 0.5
                return 0.0 if opp_history_actions[-1] == "D" else 1.0
            if strategy == "grim_forgiving":
                # Finite grim seller: punish with LQ for 2 rounds after any recent buyer boycott (D).
                return 0.0 if "D" in opp_history_actions[-2:] else 1.0
            if strategy == "noisy_hq":
                return 0.9
            if strategy == "noisy_lq":
                return 0.1
            raise ValueError(f"Unknown seller strategy for {game}: {strategy}")

        if player_idx == 2:
            if strategy == "always_buy":
                return 1.0
            if strategy == "never_buy":
                return 0.0
            if strategy == "soft_always_buy":
                return 0.9
            if strategy == "soft_never_buy":
                return 0.1
            if strategy == "tft_buy":
                if not opp_history_actions:
                    return 0.5
                return 1.0 if opp_history_actions[-1] == "HQ" else 0.0
            if strategy == "generous_buy":
                if not opp_history_actions:
                    return 1.0
                if opp_history_actions[-1] == "HQ":
                    return 1.0
                return 0.3
            if strategy == "grim_boycott":
                return 0.0 if "LQ" in opp_history_actions else 1.0
            if strategy == "grim_forgiving":
                # Finite grim buyer: punish with D for 2 rounds after any recent seller LQ.
                return 0.0 if "LQ" in opp_history_actions[-2:] else 1.0
            raise ValueError(f"Unknown buyer strategy for {game}: {strategy}")

        raise ValueError(f"Invalid player index: {player_idx}")

    raise ValueError(f"Unknown game: {game}")


def sample_from_weights(options: List[str], weights: List[float]) -> str:
    if len(options) != len(weights) or not options:
        raise ValueError("options and weights must be non-empty and same length")
    total = sum(weights)
    if total <= 0.0:
        return random.choice(options)
    threshold = random.random() * total
    running = 0.0
    for option, weight in zip(options, weights):
        running += weight
        if threshold <= running:
            return option
    return options[-1]


def sample_action_from_distribution(
    action_probs: dict[str, float],
    action_order: Tuple[str, ...] = PROMO_ACTION_ORDER,
    fallback_action: str = "Z",
) -> str:
    actions: List[str] = []
    weights: List[float] = []
    for action in action_order:
        prob = max(0.0, float(action_probs.get(action, 0.0)))
        if prob > 0.0:
            actions.append(action)
            weights.append(prob)
    if not actions:
        return fallback_action
    if len(actions) == 1:
        return actions[0]
    return sample_from_weights(actions, weights)


def action_from_prob(
    prob_primary: float,
    primary_action: str,
    secondary_action: str,
    tie_key: str,
) -> str:
    prob = min(1.0, max(0.0, prob_primary))
    if prob <= 0.0:
        return secondary_action
    if prob >= 1.0:
        return primary_action
    if abs(prob - 0.5) < 1e-12:
        # Prefer the cooperative/primary action under exact ties.
        return primary_action
    return primary_action if random.random() < prob else secondary_action


def strategy_mode_action(
    game: str,
    player_idx: int,
    strategy: str,
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> str:
    primary_action, secondary_action = binary_action_labels(game, player_idx=player_idx)
    prob_j = strategy_prob_j(game, player_idx, strategy, self_history_actions, opp_history_actions, round_idx)
    return primary_action if prob_j >= 0.5 else secondary_action


def choose_cooperative_binary_tie_strategy(
    game: str,
    player_idx: int,
    candidate_strategies: List[str],
    self_history_actions: List[str],
    opp_history_actions: List[str],
    round_idx: int,
) -> str:
    if not candidate_strategies:
        raise ValueError("candidate_strategies must be non-empty")

    if game not in ("pd", "harmony", "deadlock", "samaritan", "lemons"):
        return sorted(candidate_strategies)[0]

    scored: List[Tuple[float, str]] = []
    for strategy_name in candidate_strategies:
        coop_prob = strategy_prob_j(
            game,
            player_idx,
            strategy_name,
            self_history_actions,
            opp_history_actions,
            round_idx,
        )
        scored.append((coop_prob, strategy_name))

    best_prob = max(prob for prob, _ in scored)
    top = [name for prob, name in scored if abs(prob - best_prob) < 1e-12]
    return sorted(top)[0]


def choose_cooperative_travelers_tie_strategy(
    candidate_strategies: List[str],
    self_history_claims: List[int],
    opp_history_claims: List[int],
    round_idx: int,
    low: int,
    high: int,
) -> str:
    if not candidate_strategies:
        raise ValueError("candidate_strategies must be non-empty")

    scored: List[Tuple[int, str]] = []
    for strategy_name in candidate_strategies:
        claim = travelers_deterministic_claim(
            strategy=strategy_name,
            self_history_claims=self_history_claims,
            opp_history_claims=opp_history_claims,
            round_idx=round_idx,
            low=low,
            high=high,
        )
        scored.append((claim, strategy_name))

    best_claim = max(claim for claim, _ in scored)
    top = [name for claim, name in scored if claim == best_claim]
    return sorted(top)[0]


def strategy_log_likelihood(
    game: str,
    player_idx: int,
    strategy: str,
    realized_history: List[Tuple[object, object]],
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> float:
    if game == "travelers":
        if low is None or high is None:
            raise ValueError("low/high must be provided for Traveler's Dilemma likelihood.")
        span = high - low + 1
        if span <= 0:
            raise ValueError("Invalid Traveler's Dilemma range.")

        log_lik = 0.0
        self_claims: List[int] = []
        opp_claims: List[int] = []
        for idx, (realized_self, realized_opp) in enumerate(realized_history, start=1):
            realized_self_int = int(realized_self)
            realized_opp_int = int(realized_opp)
            if strategy == "random_claim":
                prob_realized = 1.0 / float(span)
            else:
                expected = travelers_deterministic_claim(
                    strategy,
                    self_history_claims=self_claims,
                    opp_history_claims=opp_claims,
                    round_idx=idx,
                    low=low,
                    high=high,
                )
                prob_realized = 1.0 if realized_self_int == expected else 1e-6
            prob_realized = min(1.0 - 1e-6, max(1e-6, prob_realized))
            log_lik += math.log(prob_realized)
            self_claims.append(realized_self_int)
            opp_claims.append(realized_opp_int)
        return log_lik

    if game == "promo":
        log_lik = 0.0
        self_actions: List[str] = []
        opp_actions: List[str] = []
        for idx, (realized_self, realized_opp) in enumerate(realized_history, start=1):
            probs = promo_action_probabilities(
                strategy=strategy,
                self_history_actions=self_actions,
                opp_history_actions=opp_actions,
                round_idx=idx,
            )
            realized_self_str = str(realized_self)
            prob_realized = probs.get(realized_self_str, 0.0)
            prob_realized = min(1.0 - 1e-6, max(1e-6, prob_realized))
            log_lik += math.log(prob_realized)
            self_actions.append(realized_self_str)
            opp_actions.append(str(realized_opp))
        return log_lik

    if game == "collusion":
        log_lik = 0.0
        self_actions: List[str] = []
        opp_actions: List[str] = []
        for idx, (realized_self, realized_opp) in enumerate(realized_history, start=1):
            probs = collusion_action_probabilities(
                strategy=strategy,
                self_history_actions=self_actions,
                opp_history_actions=opp_actions,
                round_idx=idx,
            )
            realized_self_str = str(realized_self)
            prob_realized = probs.get(realized_self_str, 0.0)
            prob_realized = min(1.0 - 1e-6, max(1e-6, prob_realized))
            log_lik += math.log(prob_realized)
            self_actions.append(realized_self_str)
            opp_actions.append(str(realized_opp))
        return log_lik

    primary_action, _ = binary_action_labels(game, player_idx=player_idx)
    log_lik = 0.0
    self_actions: List[str] = []
    opp_actions: List[str] = []

    for idx, (realized_self, _realized_opp) in enumerate(realized_history, start=1):
        prob_j = strategy_prob_j(game, player_idx, strategy, self_actions, opp_actions, idx)
        prob_realized = prob_j if realized_self == primary_action else (1.0 - prob_j)
        prob_realized = min(1.0 - 1e-6, max(1e-6, prob_realized))
        log_lik += math.log(prob_realized)
        self_actions.append(realized_self)
        opp_actions.append(_realized_opp)
    return log_lik


def sample_opponent_strategy(
    backend,
    game: str,
    player_idx: int,
    round_idx: int,
    rules_text: str,
    history_self_opp: List[Tuple[object, object]],
    planner_cfg: PlannerConfig,
    sample_idx: int,
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> str:
    opponent_idx = 2 if player_idx == 1 else 1
    menu = strategy_menu(game, player_idx=opponent_idx)
    collusive_prior = None
    if planner_cfg.collusive_mode:
        prior_label = collusive_prior_label(game, opponent_idx)
        if prior_label in menu:
            collusive_prior = prior_label
    label_sample_temperature = planner_cfg.sample_temperature
    # Opponent strategy should be inferred under opponent's own perspective.
    opp_view_history = [(opp_action, self_action) for self_action, opp_action in history_self_opp]
    history_start_round = 1
    inference_history = opp_view_history
    if (
        planner_cfg.strategy_memory_rounds > 0
        and len(opp_view_history) > planner_cfg.strategy_memory_rounds
    ):
        history_start_round = len(opp_view_history) - planner_cfg.strategy_memory_rounds + 1
        inference_history = opp_view_history[-planner_cfg.strategy_memory_rounds:]

    if not inference_history and not planner_cfg.collusive_mode:
        # Avoid random round-1 label flips in BOS; use a neutral memory-1 prior.
        if game == "bos" and "wsls_bos" in menu:
            return "wsls_bos"
        return random.choice(menu)

    use_likelihood = False
    if planner_cfg.strategy_inference == "llm-label":
        prompt = build_strategy_inference_prompt(
            game=game,
            round_idx=round_idx,
            rules_text=rules_text,
            history_opp_self=inference_history,
            labels=menu,
            history_start_round=history_start_round,
            collusive_prior=collusive_prior,
        )
        try:
            return backend.infer_label(
                prompt=prompt,
                labels=menu,
                sample=True,
                temperature=label_sample_temperature,
            )
        except Exception as exc:
            print(
                "Warning: strategy label inference failed; "
                f"falling back to likelihood mode. Reason: {exc.__class__.__name__}: {exc}"
            )
            use_likelihood = True
    elif planner_cfg.strategy_inference == "likelihood":
        use_likelihood = True
    else:
        raise ValueError(
            f"Unknown strategy inference mode: {planner_cfg.strategy_inference}. "
            "Supported: llm-label, likelihood"
        )

    if not use_likelihood:
        raise RuntimeError("Internal error: no strategy inference path selected.")

    log_likelihoods = [
        strategy_log_likelihood(game, opponent_idx, name, inference_history, low=low, high=high)
        for name in menu
    ]
    temp = max(planner_cfg.sample_temperature, 1e-5)
    scaled = [value / temp for value in log_likelihoods]
    max_scaled = max(scaled)
    weights = [math.exp(value - max_scaled) for value in scaled]
    sampled = sample_from_weights(menu, weights)
    if sampled in menu:
        return sampled
    tie_key = f"{game}|strategy_fallback|{round_idx}|{sample_idx}|{inference_history}"
    return deterministic_choice_from_options(tie_key, menu)


def estimate_strategy_rollout_value(
    game: str,
    player_idx: int,
    self_strategy: str,
    opp_strategy: str,
    round_idx: int,
    rounds: int,
    history_self_opp: List[Tuple[str, str]],
    planner_cfg: PlannerConfig,
    sample_idx: int,
) -> float:
    primary_action, secondary_action = binary_action_labels(game, player_idx=player_idx)
    opp_player_idx = 2 if player_idx == 1 else 1
    opp_primary_action, opp_secondary_action = binary_action_labels(game, player_idx=opp_player_idx)
    max_round = rounds
    if planner_cfg.planning_horizon > 0:
        max_round = min(rounds, round_idx + planner_cfg.planning_horizon - 1)

    sim_self_actions = [self_action for self_action, _ in history_self_opp]
    sim_opp_actions = [opp_action for _, opp_action in history_self_opp]
    value = 0.0

    for sim_round in range(round_idx, max_round + 1):
        self_prob_j = strategy_prob_j(game, player_idx, self_strategy, sim_self_actions, sim_opp_actions, sim_round)
        opp_prob_j = strategy_prob_j(
            game,
            opp_player_idx,
            opp_strategy,
            sim_opp_actions,
            sim_self_actions,
            sim_round,
        )
        self_action = action_from_prob(
            self_prob_j,
            primary_action=primary_action,
            secondary_action=secondary_action,
            tie_key=f"{game}|self|{self_strategy}|{sample_idx}|{sim_round}|{sim_self_actions}|{sim_opp_actions}",
        )
        opp_action = action_from_prob(
            opp_prob_j,
            primary_action=opp_primary_action,
            secondary_action=opp_secondary_action,
            tie_key=f"{game}|opp|{opp_strategy}|{sample_idx}|{sim_round}|{sim_opp_actions}|{sim_self_actions}",
        )
        self_points, _ = player_payoff(game, player_idx, self_action, opp_action)
        value += (planner_cfg.discount ** (sim_round - round_idx)) * float(self_points)
        sim_self_actions.append(self_action)
        sim_opp_actions.append(opp_action)

    return value


def estimate_promo_rollout_value(
    player_idx: int,
    self_strategy: str,
    opp_strategy: str,
    round_idx: int,
    rounds: int,
    history_self_opp: List[Tuple[str, str]],
    planner_cfg: PlannerConfig,
) -> float:
    max_round = rounds
    if planner_cfg.planning_horizon > 0:
        max_round = min(rounds, round_idx + planner_cfg.planning_horizon - 1)

    sim_self_actions = [self_action for self_action, _ in history_self_opp]
    sim_opp_actions = [opp_action for _, opp_action in history_self_opp]
    value = 0.0

    for sim_round in range(round_idx, max_round + 1):
        self_probs = promo_action_probabilities(
            strategy=self_strategy,
            self_history_actions=sim_self_actions,
            opp_history_actions=sim_opp_actions,
            round_idx=sim_round,
        )
        opp_probs = promo_action_probabilities(
            strategy=opp_strategy,
            self_history_actions=sim_opp_actions,
            opp_history_actions=sim_self_actions,
            round_idx=sim_round,
        )
        self_action = sample_action_from_distribution(self_probs)
        opp_action = sample_action_from_distribution(opp_probs)
        self_points, _ = player_payoff("promo", player_idx, self_action, opp_action)
        value += (planner_cfg.discount ** (sim_round - round_idx)) * float(self_points)
        sim_self_actions.append(self_action)
        sim_opp_actions.append(opp_action)

    return value


def estimate_collusion_rollout_value(
    player_idx: int,
    self_strategy: str,
    opp_strategy: str,
    round_idx: int,
    rounds: int,
    history_self_opp: List[Tuple[str, str]],
    planner_cfg: PlannerConfig,
) -> float:
    max_round = rounds
    if planner_cfg.planning_horizon > 0:
        max_round = min(rounds, round_idx + planner_cfg.planning_horizon - 1)

    sim_self_actions = [self_action for self_action, _ in history_self_opp]
    sim_opp_actions = [opp_action for _, opp_action in history_self_opp]
    value = 0.0

    for sim_round in range(round_idx, max_round + 1):
        self_probs = collusion_action_probabilities(
            strategy=self_strategy,
            self_history_actions=sim_self_actions,
            opp_history_actions=sim_opp_actions,
            round_idx=sim_round,
        )
        opp_probs = collusion_action_probabilities(
            strategy=opp_strategy,
            self_history_actions=sim_opp_actions,
            opp_history_actions=sim_self_actions,
            round_idx=sim_round,
        )
        self_action = sample_action_from_distribution(
            self_probs,
            action_order=COLLUSION_ACTION_ORDER,
            fallback_action="N",
        )
        opp_action = sample_action_from_distribution(
            opp_probs,
            action_order=COLLUSION_ACTION_ORDER,
            fallback_action="N",
        )
        self_points, _ = player_payoff("collusion", player_idx, self_action, opp_action)
        value += (planner_cfg.discount ** (sim_round - round_idx)) * float(self_points)
        sim_self_actions.append(self_action)
        sim_opp_actions.append(opp_action)

    return value


def estimate_travelers_rollout_value(
    player_idx: int,
    self_strategy: str,
    opp_strategy: str,
    round_idx: int,
    rounds: int,
    history_self_opp: List[Tuple[int, int]],
    planner_cfg: PlannerConfig,
    low: int,
    high: int,
    bonus: int,
) -> float:
    max_round = rounds
    if planner_cfg.planning_horizon > 0:
        max_round = min(rounds, round_idx + planner_cfg.planning_horizon - 1)

    sim_self_claims = [int(self_claim) for self_claim, _ in history_self_opp]
    sim_opp_claims = [int(opp_claim) for _, opp_claim in history_self_opp]
    value = 0.0

    for sim_round in range(round_idx, max_round + 1):
        self_claim = travelers_claim_from_strategy(
            strategy=self_strategy,
            self_history_claims=sim_self_claims,
            opp_history_claims=sim_opp_claims,
            round_idx=sim_round,
            low=low,
            high=high,
        )
        opp_claim = travelers_claim_from_strategy(
            strategy=opp_strategy,
            self_history_claims=sim_opp_claims,
            opp_history_claims=sim_self_claims,
            round_idx=sim_round,
            low=low,
            high=high,
        )
        self_points, _ = travelers_player_payoff(player_idx, self_claim, opp_claim, bonus)
        value += (planner_cfg.discount ** (sim_round - round_idx)) * float(self_points)
        sim_self_claims.append(self_claim)
        sim_opp_claims.append(opp_claim)

    return value


def choose_ps_br_action(
    backend,
    game: str,
    player_idx: int,
    round_idx: int,
    rounds: int,
    rules_text: str,
    history_text: str,
    history_self_opp: List[Tuple[str, str]],
    planner_cfg: PlannerConfig,
) -> Tuple[str, float, float, str, str]:
    primary_action, secondary_action = binary_action_labels(game, player_idx=player_idx)
    menu = strategy_menu(game, player_idx=player_idx)
    strategy_values: dict[str, List[float]] = {name: [] for name in menu}

    # Infer opponent strategy once per real round, then keep it fixed across
    # rollout samples for this player's decision.
    sampled_opp_strategy = sample_opponent_strategy(
        backend=backend,
        game=game,
        player_idx=player_idx,
        round_idx=round_idx,
        rules_text=rules_text,
        history_self_opp=history_self_opp,
        planner_cfg=planner_cfg,
        sample_idx=0,
    )

    for sample_idx in range(planner_cfg.samples):
        for self_strategy in menu:
            rollout_value = estimate_strategy_rollout_value(
                game=game,
                player_idx=player_idx,
                self_strategy=self_strategy,
                opp_strategy=sampled_opp_strategy,
                round_idx=round_idx,
                rounds=rounds,
                history_self_opp=history_self_opp,
                planner_cfg=planner_cfg,
                sample_idx=sample_idx,
            )
            strategy_values[self_strategy].append(rollout_value)

    avg_values = {name: sum(vals) / float(len(vals)) for name, vals in strategy_values.items()}
    best_value = max(avg_values.values())
    best_strategies = [name for name, value in avg_values.items() if abs(value - best_value) < 1e-12]
    if len(best_strategies) == 1:
        best_strategy = best_strategies[0]
    else:
        best_strategy = choose_cooperative_binary_tie_strategy(
            game=game,
            player_idx=player_idx,
            candidate_strategies=best_strategies,
            self_history_actions=[self_action for self_action, _ in history_self_opp],
            opp_history_actions=[opp_action for _, opp_action in history_self_opp],
            round_idx=round_idx,
        )

    current_self_history = [self_action for self_action, _ in history_self_opp]
    current_opp_history = [opp_action for _, opp_action in history_self_opp]
    current_prob_j = strategy_prob_j(
        game,
        player_idx,
        best_strategy,
        current_self_history,
        current_opp_history,
        round_idx,
    )
    chosen_action = action_from_prob(
        current_prob_j,
        primary_action=primary_action,
        secondary_action=secondary_action,
        tie_key=f"{game}|action_tie|{player_idx}|{round_idx}|{best_strategy}|{history_text}",
    )

    value_j_candidates: List[float] = []
    value_f_candidates: List[float] = []
    for strategy_name, value in avg_values.items():
        mode_action = strategy_mode_action(
            game,
            player_idx,
            strategy_name,
            current_self_history,
            current_opp_history,
            round_idx,
        )
        if mode_action == primary_action:
            value_j_candidates.append(value)
        else:
            value_f_candidates.append(value)

    value_j = max(value_j_candidates) if value_j_candidates else best_value
    value_f = max(value_f_candidates) if value_f_candidates else best_value
    return chosen_action, value_j, value_f, best_strategy, sampled_opp_strategy


def choose_ps_br_promo_action(
    backend,
    player_idx: int,
    round_idx: int,
    rounds: int,
    rules_text: str,
    history_text: str,
    history_self_opp: List[Tuple[str, str]],
    planner_cfg: PlannerConfig,
) -> Tuple[str, float, float, float, str, str]:
    menu = strategy_menu("promo", player_idx=player_idx)
    strategy_values: dict[str, List[float]] = {name: [] for name in menu}

    sampled_opp_strategy = sample_opponent_strategy(
        backend=backend,
        game="promo",
        player_idx=player_idx,
        round_idx=round_idx,
        rules_text=rules_text,
        history_self_opp=history_self_opp,
        planner_cfg=planner_cfg,
        sample_idx=0,
    )

    for _sample_idx in range(planner_cfg.samples):
        for self_strategy in menu:
            rollout_value = estimate_promo_rollout_value(
                player_idx=player_idx,
                self_strategy=self_strategy,
                opp_strategy=sampled_opp_strategy,
                round_idx=round_idx,
                rounds=rounds,
                history_self_opp=history_self_opp,
                planner_cfg=planner_cfg,
            )
            strategy_values[self_strategy].append(rollout_value)

    avg_values = {name: sum(vals) / float(len(vals)) for name, vals in strategy_values.items()}
    best_value = max(avg_values.values())
    best_strategies = [name for name, value in avg_values.items() if abs(value - best_value) < 1e-12]
    best_strategy = sorted(best_strategies)[0]

    current_self_history = [self_action for self_action, _ in history_self_opp]
    current_opp_history = [opp_action for _, opp_action in history_self_opp]
    current_probs = promo_action_probabilities(
        strategy=best_strategy,
        self_history_actions=current_self_history,
        opp_history_actions=current_opp_history,
        round_idx=round_idx,
    )
    chosen_action = sample_action_from_distribution(current_probs)

    value_r_candidates: List[float] = []
    value_p_candidates: List[float] = []
    value_z_candidates: List[float] = []
    for strategy_name, value in avg_values.items():
        mode_action = promo_strategy_mode_action(
            strategy=strategy_name,
            self_history_actions=current_self_history,
            opp_history_actions=current_opp_history,
            round_idx=round_idx,
        )
        if mode_action == "R":
            value_r_candidates.append(value)
        if mode_action == "P":
            value_p_candidates.append(value)
        if mode_action == "Z":
            value_z_candidates.append(value)

    value_r = max(value_r_candidates) if value_r_candidates else best_value
    value_p = max(value_p_candidates) if value_p_candidates else best_value
    value_z = max(value_z_candidates) if value_z_candidates else best_value
    return chosen_action, value_r, value_p, value_z, best_strategy, sampled_opp_strategy


def choose_ps_br_collusion_action(
    backend,
    player_idx: int,
    round_idx: int,
    rounds: int,
    rules_text: str,
    history_text: str,
    history_self_opp: List[Tuple[str, str]],
    planner_cfg: PlannerConfig,
) -> Tuple[str, float, float, float, str, str]:
    menu = strategy_menu("collusion", player_idx=player_idx)
    strategy_values: dict[str, List[float]] = {name: [] for name in menu}

    sampled_opp_strategy = sample_opponent_strategy(
        backend=backend,
        game="collusion",
        player_idx=player_idx,
        round_idx=round_idx,
        rules_text=rules_text,
        history_self_opp=history_self_opp,
        planner_cfg=planner_cfg,
        sample_idx=0,
    )

    for _sample_idx in range(planner_cfg.samples):
        for self_strategy in menu:
            rollout_value = estimate_collusion_rollout_value(
                player_idx=player_idx,
                self_strategy=self_strategy,
                opp_strategy=sampled_opp_strategy,
                round_idx=round_idx,
                rounds=rounds,
                history_self_opp=history_self_opp,
                planner_cfg=planner_cfg,
            )
            strategy_values[self_strategy].append(rollout_value)

    avg_values = {name: sum(vals) / float(len(vals)) for name, vals in strategy_values.items()}
    best_value = max(avg_values.values())
    best_strategies = [name for name, value in avg_values.items() if abs(value - best_value) < 1e-12]
    best_strategy = sorted(best_strategies)[0]

    current_self_history = [self_action for self_action, _ in history_self_opp]
    current_opp_history = [opp_action for _, opp_action in history_self_opp]
    current_probs = collusion_action_probabilities(
        strategy=best_strategy,
        self_history_actions=current_self_history,
        opp_history_actions=current_opp_history,
        round_idx=round_idx,
    )
    chosen_action = sample_action_from_distribution(
        current_probs,
        action_order=COLLUSION_ACTION_ORDER,
        fallback_action="N",
    )

    value_k_candidates: List[float] = []
    value_n_candidates: List[float] = []
    value_p_candidates: List[float] = []
    for strategy_name, value in avg_values.items():
        mode_action = collusion_strategy_mode_action(
            strategy=strategy_name,
            self_history_actions=current_self_history,
            opp_history_actions=current_opp_history,
            round_idx=round_idx,
        )
        if mode_action == "K":
            value_k_candidates.append(value)
        if mode_action == "N":
            value_n_candidates.append(value)
        if mode_action == "P":
            value_p_candidates.append(value)

    value_k = max(value_k_candidates) if value_k_candidates else best_value
    value_n = max(value_n_candidates) if value_n_candidates else best_value
    value_p = max(value_p_candidates) if value_p_candidates else best_value
    return chosen_action, value_k, value_n, value_p, best_strategy, sampled_opp_strategy


def choose_ps_br_claim(
    backend,
    player_idx: int,
    round_idx: int,
    rounds: int,
    rules_text: str,
    history_text: str,
    history_self_opp: List[Tuple[int, int]],
    planner_cfg: PlannerConfig,
    low: int,
    high: int,
    bonus: int,
) -> Tuple[int, float, float, str]:
    menu = strategy_menu("travelers")
    strategy_values: dict[str, List[float]] = {name: [] for name in menu}

    sampled_opp_strategy = sample_opponent_strategy(
        backend=backend,
        game="travelers",
        player_idx=player_idx,
        round_idx=round_idx,
        rules_text=rules_text,
        history_self_opp=history_self_opp,
        planner_cfg=planner_cfg,
        sample_idx=0,
        low=low,
        high=high,
    )

    for _sample_idx in range(planner_cfg.samples):
        for self_strategy in menu:
            rollout_value = estimate_travelers_rollout_value(
                player_idx=player_idx,
                self_strategy=self_strategy,
                opp_strategy=sampled_opp_strategy,
                round_idx=round_idx,
                rounds=rounds,
                history_self_opp=history_self_opp,
                planner_cfg=planner_cfg,
                low=low,
                high=high,
                bonus=bonus,
            )
            strategy_values[self_strategy].append(rollout_value)

    avg_values = {name: sum(vals) / float(len(vals)) for name, vals in strategy_values.items()}
    best_value = max(avg_values.values())
    best_strategies = [name for name, value in avg_values.items() if abs(value - best_value) < 1e-12]
    if len(best_strategies) == 1:
        best_strategy = best_strategies[0]
    else:
        best_strategy = choose_cooperative_travelers_tie_strategy(
            candidate_strategies=best_strategies,
            self_history_claims=[int(self_claim) for self_claim, _ in history_self_opp],
            opp_history_claims=[int(opp_claim) for _, opp_claim in history_self_opp],
            round_idx=round_idx,
            low=low,
            high=high,
        )

    current_self_claims = [int(self_claim) for self_claim, _ in history_self_opp]
    current_opp_claims = [int(opp_claim) for _, opp_claim in history_self_opp]
    chosen_claim = travelers_claim_from_strategy(
        strategy=best_strategy,
        self_history_claims=current_self_claims,
        opp_history_claims=current_opp_claims,
        round_idx=round_idx,
        low=low,
        high=high,
    )

    value_low_candidates: List[float] = []
    value_high_candidates: List[float] = []
    for strategy_name, value in avg_values.items():
        mode_claim = travelers_deterministic_claim(
            strategy=strategy_name,
            self_history_claims=current_self_claims,
            opp_history_claims=current_opp_claims,
            round_idx=round_idx,
            low=low,
            high=high,
        )
        if mode_claim == low:
            value_low_candidates.append(value)
        if mode_claim == high:
            value_high_candidates.append(value)

    value_low = max(value_low_candidates) if value_low_candidates else best_value
    value_high = max(value_high_candidates) if value_high_candidates else best_value
    return chosen_claim, value_low, value_high, best_strategy


def run_bos_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[str, str]] = []
    history_pairs_2: List[Tuple[str, str]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question_1 = BOS_PLAYER1_RULES.format(rounds=rounds)
    question_2 = BOS_PLAYER2_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        answer_1, v1j, v1f, strategy_1, sampled_opp_strategy_1 = choose_ps_br_action(
            backend_1,
            game="bos",
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question_1,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
        )
        answer_2, v2j, v2f, strategy_2, sampled_opp_strategy_2 = choose_ps_br_action(
            backend_2,
            game="bos",
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question_2,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
        )

        forced_1 = forced_first_action("bos", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("bos", player_idx=2, mode=first_action_mode) if round_idx == 1 else None
        if forced_1 is not None:
            answer_1 = forced_1
        if forced_2 is not None:
            answer_2 = forced_2

        points_1, points_2 = bos_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_history(history_1, round_idx, answer_1, answer_2, points_1, points_2)
        history_2 = append_history(history_2, round_idx, answer_2, answer_1, points_2, points_1)
        history_pairs_1.append((answer_1, answer_2))
        history_pairs_2.append((answer_2, answer_1))

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
                strategy_1,
                strategy_2,
                sampled_opp_strategy_1,
                sampled_opp_strategy_2,
                v1j,
                v1f,
                v2j,
                v2f,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_sampled_opp_strategy_p1",
            "psbr_sampled_opp_strategy_p2",
            "psbr_value_j_p1",
            "psbr_value_f_p1",
            "psbr_value_j_p2",
            "psbr_value_f_p2",
        ],
    )


def run_pd_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[str, str]] = []
    history_pairs_2: List[Tuple[str, str]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = PD_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        answer_1, v1j, v1f, strategy_1, sampled_opp_strategy_1 = choose_ps_br_action(
            backend_1,
            game="pd",
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
        )
        answer_2, v2j, v2f, strategy_2, sampled_opp_strategy_2 = choose_ps_br_action(
            backend_2,
            game="pd",
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
        )

        forced_1 = forced_first_action("pd", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("pd", player_idx=2, mode=first_action_mode) if round_idx == 1 else None
        if forced_1 is not None:
            answer_1 = forced_1
        if forced_2 is not None:
            answer_2 = forced_2

        points_1, points_2 = pd_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_history(history_1, round_idx, answer_1, answer_2, points_1, points_2)
        history_2 = append_history(history_2, round_idx, answer_2, answer_1, points_2, points_1)
        history_pairs_1.append((answer_1, answer_2))
        history_pairs_2.append((answer_2, answer_1))

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
                strategy_1,
                strategy_2,
                sampled_opp_strategy_1,
                sampled_opp_strategy_2,
                v1j,
                v1f,
                v2j,
                v2f,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_sampled_opp_strategy_p1",
            "psbr_sampled_opp_strategy_p2",
            "psbr_value_j_p1",
            "psbr_value_f_p1",
            "psbr_value_j_p2",
            "psbr_value_f_p2",
        ],
    )


def run_symmetric_binary_psbr(
    backend_1,
    backend_2,
    game: str,
    rules_text: str,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[str, str]] = []
    history_pairs_2: List[Tuple[str, str]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []
    primary_action, secondary_action = binary_action_labels(game, player_idx=1)
    primary_col = primary_action.lower()
    secondary_col = secondary_action.lower()

    for round_idx in range(1, rounds + 1):
        answer_1, v1_primary, v1_secondary, strategy_1, sampled_opp_strategy_1 = choose_ps_br_action(
            backend_1,
            game=game,
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=rules_text,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
        )
        answer_2, v2_primary, v2_secondary, strategy_2, sampled_opp_strategy_2 = choose_ps_br_action(
            backend_2,
            game=game,
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=rules_text,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
        )

        forced_1 = forced_first_action(game, player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action(game, player_idx=2, mode=first_action_mode) if round_idx == 1 else None
        if forced_1 is not None:
            answer_1 = forced_1
        if forced_2 is not None:
            answer_2 = forced_2

        points_1, points_2 = stage_points(game, answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_history(history_1, round_idx, answer_1, answer_2, points_1, points_2)
        history_2 = append_history(history_2, round_idx, answer_2, answer_1, points_2, points_1)
        history_pairs_1.append((answer_1, answer_2))
        history_pairs_2.append((answer_2, answer_1))

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
                strategy_1,
                strategy_2,
                sampled_opp_strategy_1,
                sampled_opp_strategy_2,
                v1_primary,
                v1_secondary,
                v2_primary,
                v2_secondary,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_sampled_opp_strategy_p1",
            "psbr_sampled_opp_strategy_p2",
            f"psbr_value_{primary_col}_p1",
            f"psbr_value_{secondary_col}_p1",
            f"psbr_value_{primary_col}_p2",
            f"psbr_value_{secondary_col}_p2",
        ],
    )


def run_harmony_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    question = HARMONY_RULES.format(rounds=rounds)
    return run_symmetric_binary_psbr(
        backend_1=backend_1,
        backend_2=backend_2,
        game="harmony",
        rules_text=question,
        model_name=model_name,
        rounds=rounds,
        planner_cfg=planner_cfg,
        first_action_mode=first_action_mode,
    )


def run_deadlock_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    question = DEADLOCK_RULES.format(rounds=rounds)
    return run_symmetric_binary_psbr(
        backend_1=backend_1,
        backend_2=backend_2,
        game="deadlock",
        rules_text=question,
        model_name=model_name,
        rounds=rounds,
        planner_cfg=planner_cfg,
        first_action_mode=first_action_mode,
    )


def run_promo_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[str, str]] = []
    history_pairs_2: List[Tuple[str, str]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = PROMO_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        answer_1, v1r, v1p, v1z, strategy_1, sampled_opp_strategy_1 = choose_ps_br_promo_action(
            backend=backend_1,
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
        )
        answer_2, v2r, v2p, v2z, strategy_2, sampled_opp_strategy_2 = choose_ps_br_promo_action(
            backend=backend_2,
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
        )

        forced_1 = forced_first_action("promo", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("promo", player_idx=2, mode=first_action_mode) if round_idx == 1 else None
        if forced_1 is not None:
            answer_1 = forced_1
        if forced_2 is not None:
            answer_2 = forced_2

        points_1, points_2 = promo_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_history(history_1, round_idx, answer_1, answer_2, points_1, points_2)
        history_2 = append_history(history_2, round_idx, answer_2, answer_1, points_2, points_1)
        history_pairs_1.append((answer_1, answer_2))
        history_pairs_2.append((answer_2, answer_1))

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
                strategy_1,
                strategy_2,
                sampled_opp_strategy_1,
                sampled_opp_strategy_2,
                v1r,
                v1p,
                v1z,
                v2r,
                v2p,
                v2z,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_sampled_opp_strategy_p1",
            "psbr_sampled_opp_strategy_p2",
            "psbr_value_r_p1",
            "psbr_value_p_p1",
            "psbr_value_z_p1",
            "psbr_value_r_p2",
            "psbr_value_p_p2",
            "psbr_value_z_p2",
        ],
    )


def run_collusion_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[str, str]] = []
    history_pairs_2: List[Tuple[str, str]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = COLLUSION_RULES.format(rounds=rounds)

    for round_idx in range(1, rounds + 1):
        answer_1, v1k, v1n, v1p, strategy_1, sampled_opp_strategy_1 = choose_ps_br_collusion_action(
            backend=backend_1,
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
        )
        answer_2, v2k, v2n, v2p, strategy_2, sampled_opp_strategy_2 = choose_ps_br_collusion_action(
            backend=backend_2,
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
        )

        forced_1 = forced_first_action("collusion", player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action("collusion", player_idx=2, mode=first_action_mode) if round_idx == 1 else None
        if forced_1 is not None:
            answer_1 = forced_1
        if forced_2 is not None:
            answer_2 = forced_2

        points_1, points_2 = collusion_points(answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_history(history_1, round_idx, answer_1, answer_2, points_1, points_2)
        history_2 = append_history(history_2, round_idx, answer_2, answer_1, points_2, points_1)
        history_pairs_1.append((answer_1, answer_2))
        history_pairs_2.append((answer_2, answer_1))

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
                strategy_1,
                strategy_2,
                sampled_opp_strategy_1,
                sampled_opp_strategy_2,
                v1k,
                v1n,
                v1p,
                v2k,
                v2n,
                v2p,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_sampled_opp_strategy_p1",
            "psbr_sampled_opp_strategy_p2",
            "psbr_value_k_p1",
            "psbr_value_n_p1",
            "psbr_value_p_p1",
            "psbr_value_k_p2",
            "psbr_value_n_p2",
            "psbr_value_p_p2",
        ],
    )


def run_asymmetric_binary_psbr(
    backend_1,
    backend_2,
    game: str,
    rules_text_p1: str,
    rules_text_p2: str,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[str, str]] = []
    history_pairs_2: List[Tuple[str, str]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    p1_primary_action, p1_secondary_action = binary_action_labels(game, player_idx=1)
    p2_primary_action, p2_secondary_action = binary_action_labels(game, player_idx=2)

    for round_idx in range(1, rounds + 1):
        answer_1, v1_primary, v1_secondary, strategy_1, sampled_opp_strategy_1 = choose_ps_br_action(
            backend_1,
            game=game,
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=rules_text_p1,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
        )
        answer_2, v2_primary, v2_secondary, strategy_2, sampled_opp_strategy_2 = choose_ps_br_action(
            backend_2,
            game=game,
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=rules_text_p2,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
        )

        forced_1 = forced_first_action(game, player_idx=1, mode=first_action_mode) if round_idx == 1 else None
        forced_2 = forced_first_action(game, player_idx=2, mode=first_action_mode) if round_idx == 1 else None
        if forced_1 is not None:
            answer_1 = forced_1
        if forced_2 is not None:
            answer_2 = forced_2

        points_1, points_2 = stage_points(game, answer_1, answer_2)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_history(history_1, round_idx, answer_1, answer_2, points_1, points_2)
        history_2 = append_history(history_2, round_idx, answer_2, answer_1, points_2, points_1)
        history_pairs_1.append((answer_1, answer_2))
        history_pairs_2.append((answer_2, answer_1))

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
                strategy_1,
                strategy_2,
                sampled_opp_strategy_1,
                sampled_opp_strategy_2,
                v1_primary,
                v1_secondary,
                v2_primary,
                v2_secondary,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_sampled_opp_strategy_p1",
            "psbr_sampled_opp_strategy_p2",
            f"psbr_value_{p1_primary_action.lower()}_p1",
            f"psbr_value_{p1_secondary_action.lower()}_p1",
            f"psbr_value_{p2_primary_action.lower()}_p2",
            f"psbr_value_{p2_secondary_action.lower()}_p2",
        ],
    )


def run_samaritan_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    question_1 = SAMARITAN_HELPER_RULES.format(rounds=rounds)
    question_2 = SAMARITAN_RECIPIENT_RULES.format(rounds=rounds)
    return run_asymmetric_binary_psbr(
        backend_1=backend_1,
        backend_2=backend_2,
        game="samaritan",
        rules_text_p1=question_1,
        rules_text_p2=question_2,
        model_name=model_name,
        rounds=rounds,
        planner_cfg=planner_cfg,
        first_action_mode=first_action_mode,
    )


def run_lemons_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    first_action_mode: str,
) -> pd.DataFrame:
    question_1 = LEMONS_SELLER_RULES.format(rounds=rounds)
    question_2 = LEMONS_BUYER_RULES.format(rounds=rounds)
    return run_asymmetric_binary_psbr(
        backend_1=backend_1,
        backend_2=backend_2,
        game="lemons",
        rules_text_p1=question_1,
        rules_text_p2=question_2,
        model_name=model_name,
        rounds=rounds,
        planner_cfg=planner_cfg,
        first_action_mode=first_action_mode,
    )


def run_travelers_psbr(
    backend_1,
    backend_2,
    model_name: str,
    rounds: int,
    planner_cfg: PlannerConfig,
    low: int,
    high: int,
    bonus: int,
) -> pd.DataFrame:
    history_1 = ""
    history_2 = ""
    history_pairs_1: List[Tuple[int, int]] = []
    history_pairs_2: List[Tuple[int, int]] = []
    total_1 = 0
    total_2 = 0
    rows: List[List[object]] = []

    question = travelers_rules(rounds=rounds, low=low, high=high, bonus=bonus)

    for round_idx in range(1, rounds + 1):
        claim_1, v1_low, v1_high, strategy_1 = choose_ps_br_claim(
            backend=backend_1,
            player_idx=1,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_1,
            history_self_opp=history_pairs_1,
            planner_cfg=planner_cfg,
            low=low,
            high=high,
            bonus=bonus,
        )
        claim_2, v2_low, v2_high, strategy_2 = choose_ps_br_claim(
            backend=backend_2,
            player_idx=2,
            round_idx=round_idx,
            rounds=rounds,
            rules_text=question,
            history_text=history_2,
            history_self_opp=history_pairs_2,
            planner_cfg=planner_cfg,
            low=low,
            high=high,
            bonus=bonus,
        )

        points_1, points_2 = travelers_points(claim_1, claim_2, bonus=bonus)
        total_1 += points_1
        total_2 += points_2

        history_1 = append_travelers_history(history_1, round_idx, claim_1, claim_2, points_1, points_2)
        history_2 = append_travelers_history(history_2, round_idx, claim_2, claim_1, points_2, points_1)
        history_pairs_1.append((claim_1, claim_2))
        history_pairs_2.append((claim_2, claim_1))

        rows.append(
            [
                round_idx,
                model_name,
                model_name,
                claim_1,
                claim_2,
                points_1,
                points_2,
                total_1,
                total_2,
                strategy_1,
                strategy_2,
                v1_low,
                v1_high,
                v2_low,
                v2_high,
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
            "psbr_strategy_p1",
            "psbr_strategy_p2",
            "psbr_value_low_p1",
            "psbr_value_high_p1",
            "psbr_value_low_p2",
            "psbr_value_high_p2",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated games with a PS-BR-style planner.")
    parser.add_argument(
        "--backend",
        choices=["hf-local", "openai-compatible", "mock"],
        default="hf-local",
        help="Inference backend used for strategy-label inference when enabled.",
    )
    parser.add_argument(
        "--game",
        choices=["bos", "pd", "deadlock", "promo", "collusion", "samaritan", "lemons", "both", "all"],
        default="both",
        help="Which game to run.",
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name used for inference and outputs.")
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per repeated game.")
    parser.add_argument(
        "--bos-output",
        default="ps_br/experiment_bos_psbr_gpt_oss_20b.csv",
        help="Output CSV path for BOS.",
    )
    parser.add_argument(
        "--pd-output",
        default="ps_br/experiment_pd_psbr_gpt_oss_20b.csv",
        help="Output CSV path for PD.",
    )
    parser.add_argument(
        "--deadlock-output",
        default="ps_br/experiment_deadlock_psbr_gpt_oss_20b.csv",
        help="Output CSV path for Deadlock.",
    )
    parser.add_argument(
        "--promo-output",
        default="ps_br/experiment_promo_psbr_gpt_oss_20b.csv",
        help="Output CSV path for Promo.",
    )
    parser.add_argument(
        "--collusion-output",
        default="ps_br/experiment_collusion_psbr_gpt_oss_20b.csv",
        help="Output CSV path for Collusion.",
    )
    parser.add_argument(
        "--samaritan-output",
        default="ps_br/experiment_samaritan_psbr_gpt_oss_20b.csv",
        help="Output CSV path for Samaritan.",
    )
    parser.add_argument(
        "--lemons-output",
        default="ps_br/experiment_lemons_psbr_gpt_oss_20b.csv",
        help="Output CSV path for Lemons.",
    )
    parser.add_argument(
        "--first-action-mode",
        choices=FIRST_ACTION_CHOICES,
        default="model",
        help=(
            "Round-1 action policy. 'model' uses PS-BR output. "
            "'defect' uses self-favoring one-shot actions for each game."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--ps-samples",
        type=int,
        default=4,
        help="Number of sampled strategy-level continuation rollouts.",
    )
    parser.add_argument(
        "--planning-horizon",
        type=int,
        default=10,
        help="Lookahead horizon in rounds. 0 means full remaining horizon.",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=1.0,
        help="Discount factor for continuation value estimation.",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature for strategy inference and rollout sampling. "
            "Default is 0.3 for all games."
        ),
    )
    parser.add_argument(
        "--strategy-inference",
        choices=["llm-label", "likelihood"],
        default="llm-label",
        help="How to infer opponent strategy from history.",
    )
    parser.add_argument(
        "--strategy-memory-rounds",
        type=int,
        default=0,
        help="Opponent strategy inference memory window in rounds. 0 means full history.",
    )
    parser.add_argument(
        "--collusive-mode",
        action="store_true",
        help=(
            "Enable collusive prior guidance in strategy-label inference prompts for BOS/PD/Promo/Collusion/Samaritan/Lemons. "
            "When enabled, round-1 opponent inference uses normal inference flow (no random fallback shortcut)."
        ),
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
        default=4,
        help="Max generated tokens per model call.",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=2,
        help="How many times to re-prompt when the model output cannot be parsed.",
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

    if args.ps_samples <= 0:
        raise SystemExit("--ps-samples must be >= 1")
    if not (0.0 <= args.discount <= 1.0):
        raise SystemExit("--discount must be in [0, 1]")
    if args.strategy_memory_rounds < 0:
        raise SystemExit("--strategy-memory-rounds must be >= 0")

    if args.game == "both":
        selected_games = ["bos", "pd"]
    elif args.game == "all":
        selected_games = ["bos", "pd", "deadlock", "promo", "collusion", "samaritan", "lemons"]
    else:
        selected_games = [args.game]

    sample_temperature = args.sample_temperature
    if sample_temperature is None:
        sample_temperature = 0.3

    random.seed(args.seed)

    if args.strategy_inference == "likelihood":
        if args.backend != "mock":
            print(
                "Note: --strategy-inference likelihood selected; "
                "model backend is not used for strategy inference."
            )
        backend_1 = MockBackend(seed=args.seed)
        backend_2 = backend_1
    elif args.backend == "openai-compatible":
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

    planner_cfg = PlannerConfig(
        samples=args.ps_samples,
        planning_horizon=args.planning_horizon,
        discount=args.discount,
        sample_temperature=sample_temperature,
        strategy_inference=args.strategy_inference,
        strategy_memory_rounds=args.strategy_memory_rounds,
        collusive_mode=args.collusive_mode,
    )

    def add_planner_metadata(frame: pd.DataFrame) -> pd.DataFrame:
        frame["psbr_samples"] = planner_cfg.samples
        frame["psbr_horizon"] = planner_cfg.planning_horizon
        frame["psbr_discount"] = planner_cfg.discount
        frame["psbr_sample_temperature"] = planner_cfg.sample_temperature
        frame["psbr_strategy_inference"] = planner_cfg.strategy_inference
        frame["psbr_strategy_memory_rounds"] = planner_cfg.strategy_memory_rounds
        frame["psbr_collusive_mode"] = planner_cfg.collusive_mode
        return frame

    if "bos" in selected_games:
        bos_df = run_bos_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        bos_df = add_planner_metadata(bos_df)
        ensure_parent_dir(args.bos_output)
        bos_df.to_csv(args.bos_output, index=False)
        print(f"Wrote BOS PS-BR results to {args.bos_output}")

    if "pd" in selected_games:
        pd_df = run_pd_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        pd_df = add_planner_metadata(pd_df)
        ensure_parent_dir(args.pd_output)
        pd_df.to_csv(args.pd_output, index=False)
        print(f"Wrote PD PS-BR results to {args.pd_output}")

    if "deadlock" in selected_games:
        deadlock_df = run_deadlock_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        deadlock_df = add_planner_metadata(deadlock_df)
        ensure_parent_dir(args.deadlock_output)
        deadlock_df.to_csv(args.deadlock_output, index=False)
        print(f"Wrote Deadlock PS-BR results to {args.deadlock_output}")

    if "promo" in selected_games:
        promo_df = run_promo_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        promo_df = add_planner_metadata(promo_df)
        ensure_parent_dir(args.promo_output)
        promo_df.to_csv(args.promo_output, index=False)
        print(f"Wrote Promo PS-BR results to {args.promo_output}")

    if "collusion" in selected_games:
        collusion_df = run_collusion_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        collusion_df = add_planner_metadata(collusion_df)
        ensure_parent_dir(args.collusion_output)
        collusion_df.to_csv(args.collusion_output, index=False)
        print(f"Wrote Collusion PS-BR results to {args.collusion_output}")

    if "samaritan" in selected_games:
        samaritan_df = run_samaritan_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        samaritan_df = add_planner_metadata(samaritan_df)
        ensure_parent_dir(args.samaritan_output)
        samaritan_df.to_csv(args.samaritan_output, index=False)
        print(f"Wrote Samaritan PS-BR results to {args.samaritan_output}")

    if "lemons" in selected_games:
        lemons_df = run_lemons_psbr(
            backend_1,
            backend_2,
            args.model,
            args.rounds,
            planner_cfg,
            first_action_mode=args.first_action_mode,
        )
        lemons_df = add_planner_metadata(lemons_df)
        ensure_parent_dir(args.lemons_output)
        lemons_df.to_csv(args.lemons_output, index=False)
        print(f"Wrote Lemons PS-BR results to {args.lemons_output}")


if __name__ == "__main__":
    main()
