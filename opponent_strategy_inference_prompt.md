# Opponent Strategy Inference Prompt

Use this template to infer one opponent strategy label from repeated-game history.

```text
You are inferring the opponent strategy in repeated {game_name}.
{rules_text}
Observed rounds so far: {observed_rounds}.

Allowed labels:
{menu_lines}

Observed action history tuple format: (opponent action, your action).
In each round line, 'opp' is the opponent action and 'you' is your action.
Infer the opponent strategy in two steps:
1) Internally infer opponent's behavior type from recent history:
- history-independent (mostly the same action regardless of your past actions),
- reactive (depends on your previous actions),
- periodic (alternating/cycle),
- or a noisy variant of one of the above.
2) Map that inferred behavior to exactly one allowed label.
Evaluate consistency over all observed rounds, with extra weight on recent rounds.
Choose the label that best explains and predicts the opponent actions across the history.
If multiple labels are close, choose the one with better predictive fit on recent rounds.
The opponent may change strategy over time; if you detect a shift, prioritize the
most recent consistent behavior.
{history_lines}

Internally assign a compatibility score from 0 to 100 to every allowed label, convert
them into relative posterior weights, and sample exactly one final label from those weights.
Output rule: do NOT output scores, reasoning, or ranking.
Respond with exactly one label only.

**Output only the label.**
```

`{history_lines}` should be one line per round, for example:

```text
Round 1: opp=J, you=F
Round 2: opp=J, you=J
Round 3: opp=F, you=J
```
