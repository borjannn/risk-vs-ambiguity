# Task 2 follow-up: behavioral model pilot

To reproduce: `uv run scripts/pilot.py` (resumes from `data/processed/pilot_raw.jsonl`, writes `results/pilot_results.json`). `uv run scripts/pilot.py --analyse-only` reanalyses the cached responses without sending requests. `uv run scripts/ellsberg_demo.py` runs the two-urn question, and `uv run scripts/export_answers.py` writes every question and answer to [`pilot_answers.csv`](pilot_answers.csv).

## Status: complete

Both models answered all 100 problems, in all 3 versions and both orders (1,200 answers), plus 40 repeat requests and the 8 Ellsberg prompts. No answer was unreadable.

## Design

- **Problems:** 100 choices13k ambiguous problems, each in the `ambiguous`, `true` and `uniform` conditions. They come from the 1,946 with feedback-only human data and non-uniform true probabilities, keeping the 50% share of two-outcome Bs. 52 have a sure-thing Option A. **The 452 human-comparison problems are untouched.**
- **Exclude them later:** the IDs are in `problem_ids` in `pilot_results.json`. Leave them out of the confirmatory analyses.
- **Prompt (`v2_chat`):** the API has no raw completion route, so `Answer:` moves from the user message to an assistant prefill, with thinking turned off.
- **Reading a choice:** P(B) = P(B) / (P(A) + P(B)) from the first token's top 20 logprobs. Negative gaps mean the model chooses the tested option (the original Option B) less often.
- **Label swap:** every stimulus also runs with the original B labeled A and listed first. The main results average both orders, which cancels a bias toward a label or a position. The swap changes both together, so it can't tell them apart.

## Results, averaged over both orders

| | Gemma | Qwen |
|---|---|---|
| Tested option chosen, of 200 answers: ambiguous / true / uniform | 52 / 88 / 92 | 34 / 57 / 96 |
| Mean P(tested option): ambiguous / true / uniform | 0.28 / 0.44 / 0.46 | 0.32 / 0.41 / 0.49 |
| **Gap, ambiguous − uniform** | **−18 pt** [−23, −13] | **−17 pt** [−19, −14] |
| **Gap, ambiguous − true** | **−16 pt** [−21, −12] | **−9 pt** [−12, −5] |
| Gap, true − uniform | −2 pt [−7, 3] | −8 pt [−12, −4] |
| Logit gap, ambiguous − uniform | −3.7 [−4.6, −2.9] | −0.79 [−0.90, −0.67] |
| Logit gap, ambiguous − true | −2.5 [−3.3, −1.6] | −0.46 [−0.63, −0.29] |
| Problems with lower P on ambiguous than uniform | 77 of 100 | 93 of 100 |
| Chooses the higher-EV option, `true` / `uniform` | 76% / 79% | 72% / 76% |
| Spearman of ambiguous logit with uniform / true logit | 0.63 / 0.48 | 0.75 / 0.16 |

The brackets are bootstrap 95% CIs over problems. The discrete counts use the token each model returned. An exact A/B tie (34 of Qwen's 600 answers, 4 of Gemma's) always came back as A, so the P rows are the safer measure.

The `true` twins have a much lower EV for B than the `uniform` twins in this sample (mean A 12.9, B true 11.5, B uniform 22.5), and B beats A on 37% vs. 67% of problems. The ambiguous − uniform gap is therefore the cleaner measure of aversion beyond a uniform prior. The ambiguous − true gap mixes aversion with this EV difference.

## The label swap changed the picture for Gemma

| | Gemma, original | Gemma, swapped | Qwen, original | Qwen, swapped |
|---|---|---|---|---|
| Ambiguous − true | −31 pt [−40, −24] | **−2 pt [−6, 2]** | −9 pt [−14, −4] | −9 pt [−12, −5] |
| Ambiguous − uniform | −26 pt [−34, −17] | −11 pt [−16, −6] | −18 pt [−22, −14] | −15 pt [−18, −13] |
| Logit gap, ambiguous − true | −3.4 [−4.4, −2.4] | −1.5 [−2.5, −0.6] | −0.51 [−0.74, −0.28] | −0.41 [−0.58, −0.26] |
| Logit gap, ambiguous − uniform | −4.2 [−5.2, −3.1] | −3.2 [−4.1, −2.4] | −0.86 [−1.02, −0.69] | −0.72 [−0.83, −0.61] |
| Tested option chosen, of 100: ambiguous / true / uniform | 49 / 84 / 78 | 3 / 4 / 14 | 12 / 21 / 44 | 22 / 36 / 52 |

- **Gemma strongly favors whichever option is labeled B and listed last.** The bias is about 5–6 logits (P(label B) = 0.74–0.89), so its `true` answers picked the same option in both orders on only 20 of 100 problems. The earlier −31 pt was largely inflated by this. In the swapped order nearly everything sits at the floor, which hides the gap on the probability scale, though it is still there on the logit scale.
- **Qwen barely reacts to the swap.** Its bias is −0.15 to −0.22 logits, and its gaps agree across the orders. It's the more trustworthy of the two on this prompt format.
- **Averaging also removes the apparent EV blindness.** The share choosing the higher EV on `true` rises from 53% to 76% for Gemma and from 64% to 72% for Qwen.

## Ellsberg two-urn question

Each model chose between a 50/50 urn and an unknown-ratio urn, betting on red and on black, with each urn in each position (4 prompts per model). Both models chose the known urn **4 of 4** times, for both colors and in both positions. Mean P(known urn) was 0.997 for Gemma and 0.949 for Qwen. Choosing it for both colors is the Ellsberg pattern, because no single belief about the unknown urn justifies it. Gemma's label bias doesn't override this. It's one wording of a famous problem, so it's a demonstration, not an estimate (see Task 13).

## What this means for the hypotheses

- **Aversion direction:** both models avoid the unknown option, in both orders, on both the gap and the Ellsberg question. This is the robust finding.
- **H3B, behaviorally:** a plain uniform prior predicts ambiguous = uniform, and the models sit 17–18 pt below that. Their choices do rank more like the uniform twin than the true twin, which looks like a prior plus extra avoidance. That ranking is partly expected, though, because the true probabilities aren't in the ambiguous prompt. The real H3B test is the probability readout from activations, which hasn't been run.
- **H3A, H3C, H3D:** not tested. This pilot has no activations and no human comparison, and it can't separate ambiguity from the word "unknown".

## Exploratory patterns, unconfirmed

- **Option A type:** Gemma's ambiguous − uniform gap is −1 pt [−6, 3] when A is a sure thing but −36 pt [−42, −30] when A is a gamble. Qwen shows no such split (−17 pt [−20, −13] vs. −16 pt [−20, −13]). The Ellsberg question is gamble vs. gamble, so it can't separate the two models here.
- **Number of outcomes:** Qwen's ambiguous − true gap is −20 pt [−24, −15] with two-outcome Bs and +2 pt [−2, 6] with more outcomes. Its ambiguous − uniform gap holds in both (−22 pt and −11 pt). Gemma's gaps also shrink with more outcomes.

## What this means for power

The human SD placeholder (0.11) was too small. Averaged over both orders, the model's SD is about **2.3× larger for Gemma** on both gaps, and for Qwen **1.7× larger on ambiguous − true and 1.1× larger on ambiguous − uniform**. These figures assume the main analyses also average both orders, which doubles the number of prompts to run. The Task 2 layer correction is still at the placeholder 42 layers.

| Test | Task 2 at SD 0.11 | Gemma, ambig. − true (SD 0.25) | Qwen, ambig. − true (SD 0.19) | Gemma, ambig. − uniform (SD 0.26) | Qwen, ambig. − uniform (SD 0.13) |
|---|---|---|---|---|---|
| Behavioral effect, 452 problems | 1.5 pt | 3.3 pt | 2.5 pt | 3.4 pt | 1.7 pt |
| Behavioral effect, 2,413 problems | 0.6 pt | 1.4 pt | 1.1 pt | 1.5 pt | 0.7 pt |
| Ablation minimum gap, any closure, 2,413 problems | 2.8 pt | 6.3 pt | 4.8 pt | 6.4 pt | 3.1 pt |
| Ablation minimum gap, lower bound ≥ ⅓ if true closure is ½, 2,413 problems | not computed | 12.5 pt | **9.5 pt** | 12.8 pt | 6.3 pt |
| Same lower-bound rule, 11,922 twins | not computed | 5.6 pt | 4.3 pt | 5.8 pt | 2.8 pt |

**The GO verdict still holds for behavior.** All the measured gaps clear the behavioral thresholds. For ablation, the only miss at 2,413 problems is Qwen's −9 pt ambiguous − true gap against 9.5 pt, which clears with all the twins. Using the gap's SD as the SD of the ablation shift is an assumption, because no ablation has been run. Gemma's saturation (57% of original-order answers below .01 or above .99) also makes its gaps far from normal, so the analysis should report the logit scale as well.

## What to resolve before reading the gaps as ambiguity aversion

1. **The word "unknown".** The gap could be a reaction to that word or to the shape of the prompt rather than to missing probabilities. This is what the hedged-language distractors in Task 13 and H3D test.
2. **Token length.** Ambiguous prompts have **about 15 fewer tokens** than `true` prompts (SD 14), although they have more characters. The difference grows with the number of outcomes, so it needs matching or a covariate in Task 12.
3. **Prompt format.** Only one template has been used here, and the first test prompt moved Qwen from P(B) = 0.65 to 0.13 when the prefill was added. An alternative template is still needed as a robustness check. Gemma's label bias may also depend on this format.
4. **Measurement noise.** Qwen's logits all sit on multiples of 1/8, and many of Gemma's do (42%). In 20 repeat requests per model, Qwen never changed, but one Gemma answer moved by 1.5 logits. That's too few to size this.
5. **Human comparison.** None has been made. Humans chose the ambiguous B 54% of the time on these 100 problems, but only after feedback from repeated play and without twins, so this isn't comparable to the models' answers.

## Scope and environment

This is behavior only, through a hosted API. It gives no activations, and the hosted weights may be quantized, so these numbers carry over to the mechanistic tasks only if the same weights can be run locally. All the pilot problems were used to look for patterns, so everything here is exploratory, and the confirmatory analyses need the other problems. `uv` isn't installed on this machine, so the run used system Python (pandas 3.0.5, older than the `pyproject.toml` minimum). The rebuilt stimuli reproduced the committed summary exactly.

The API key was rejected several times during the run (HTTP 401), with the same key in `.env` working before and after some of the failures, and a new key finished the run. The cause isn't known. Answers already cached were kept.
