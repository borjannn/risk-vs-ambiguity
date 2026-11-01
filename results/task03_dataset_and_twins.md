# Task 3: Main dataset and the twin stimulus set

To reproduce: `uv run scripts/build_stimuli.py`. It writes `data/processed/stimuli.jsonl` (32 MB, gitignored) and `results/stimuli_summary.json`.

## Decisions

| Dataset | Role | Why (task 2) |
|---|---|---|
| **choices13k** | **Main dataset** | 2,513 ambiguous problems for the mechanistic tests (MDE d = 0.06, AUC CI ± 0.013) and 452 with no-feedback human rates for the human comparison (MDE r = 0.19). |
| **CPC18** | **Replication of the behavioral and human comparison** | Very reliable human rates (ceiling 0.96) and real human aversion beyond the prior (−7 pt), but 40 ambiguous games are too few for the mechanistic decision rules. |
| **Synthetic Ellsberg set** | **Supporting**: H3D specificity probe | choices13k has enough items, so the synthetic set doesn't need to carry the main analyses. It's still needed for the hedged-language distractors and the urn and loss framings, which the datasets don't have. |

## What was built

**Every problem in both datasets** is rendered in up to three conditions that differ *only* in how Option B's probabilities are described. Option A is always fully described.

| Condition | Option B shows | Used for |
|---|---|---|
| `ambiguous` | B's outcomes, each "with probability unknown" | the ambiguity side of every contrast |
| `true` | B's real probabilities (the hidden ones, for ambiguous problems) | total gap; H3A diff-of-means; same EV and outcomes as the ambiguous version |
| `uniform` | each of B's k outcomes at 1/k | H3B: the model should treat `ambiguous` like this if it assumes a uniform prior; the gap against this version isolates aversion beyond the prior |

You asked to keep the extra data, so twins were built for **all** problems, not only the ones humans saw as ambiguous. A problem that humans saw as risky still gets an `ambiguous` version. That version has no human data, but it adds matched pairs for the mechanistic analyses.

`human_condition = true` marks the row showing what participants actually saw. Only those rows carry human rates, with no-feedback and feedback rates kept separate (which also covers task 11).

| | choices13k | CPC18 |
|---|---|---|
| Problems | 13,006 | 270 |
| … with all three conditions | 12,751 | 265 |
| … `true` only (B has a single outcome, so there's nothing to hide) | 255 | 5 |
| Humans saw ambiguous | 2,513 | 40 |
| … with no-feedback human rate | 452 | 40 |
| Uniform twin identical to true twin (true probabilities already 1/k) | 729 | 26 |
| **Total stimuli** | **38,508** | **800** |

### Example (choices13k problem 2, which humans saw as ambiguous)

```
Option A: 14 with probability 60%, -18 with probability 40%
Option B: 8 with probability unknown, -5 with probability unknown    ← ambiguous
Option B: 8 with probability 25%, -5 with probability 75%            ← true
Option B: 8 with probability 50%, -5 with probability 50%            ← uniform
```

The full prompt (`v1`):

```
You are choosing between two options.
Option A: {A}
Option B: {B}
Which option do you choose? Answer with A or B.
Answer:
```

Outcomes are listed in descending order, so all three versions of a problem line up word for word. Probabilities are shown as percentages with up to 4 significant digits.

## Balance (for task 12)

- **EV and outcome range are exactly matched** across `ambiguous` and `true` by construction: same outcomes, same hidden distribution.
- `uniform` deliberately changes B's *stated* EV, by 14.7 points on average in choices13k.
- **Prompt length:** ambiguous prompts are about 10 characters longer (250.5 vs. 240.6 in choices13k), because "unknown" is longer than "25%". The phrase was chosen to keep the frame identical ("with probability ___"). The first draft, "with an unknown probability", was 21 characters longer. **Token-level matching has to wait for the model's tokenizer.** Once the model is chosen, count tokens per condition and either confirm they're balanced or add token length as a covariate.

## Known limitations and open choices

- **The prompt wording is a choice**, not the wording participants saw: choices13k and CPC18 used graphical layouts. Steering results are known to be sensitive to prompt format, so plan one alternative template as a robustness check. That's easy to do: change `PROMPT` and bump `PROMPT_VERSION`.
- **The human rate is attached only to the condition humans saw.** For problems humans saw as risky, the `ambiguous` and `uniform` rows have no human comparison.
- **The 729 + 26 problems whose true probabilities are already uniform** (`uniform_equals_true`) make the `true` and `uniform` versions identical. Exclude them from analyses that compare those two versions.
