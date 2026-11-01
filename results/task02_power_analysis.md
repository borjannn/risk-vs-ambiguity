# Task 2: Power analysis

*Assigned to Andrej; run by Borjan.* To reproduce: `uv run scripts/power_analysis.py` (writes `results/power_analysis.json`).

The number of problems is fixed by the datasets (task 1), so this is a **sensitivity analysis**. For each test and each subset, it gives the **minimum detectable effect (MDE)** at α = .05 and 80% power. Tests that run at every layer are also Bonferroni-corrected across 42 layers (Gemma-2-9B, the most conservative case). FDR correction will land between the two.

## Verdict: GO, with choices13k as the main dataset

| Test | choices13k | CPC18 | Verdict |
|---|---|---|---|
| **Behavioral baseline**: does the model avoid ambiguity? (paired, ambiguous vs. disclosed twin) | MDE ≈ **0.6 pt** (all 2,513) / 1.5 pt (452 no-feedback) | MDE ≈ 6 pt (40) | ✅ choices13k · ⚠️ CPC18 only detects large effects |
| **Model vs. human**, item-level correlation of the ambiguity gap | MDE **r = 0.19** (452; corrected for human noise) | r = 0.45 (40) | ✅ choices13k · ⚠️ CPC18 large effects only |
| **H3A**: held-out AUC ≥ 0.75 | 95% CI ± **0.013** (2,513) | ± 0.107 (40) | ✅ choices13k · ❌ CPC18 can't tell 0.75 from 0.65 |
| **H3B**: readout within ±0.05 of 0.5 (TOST) | works if readout SD ≤ **0.42** (1,348 two-outcome, layer-corrected) | needs SD ≤ 0.054 (26) | ✅ choices13k · ❌ CPC18 |
| **Ablation** closes ≥ ⅓ of the gap | detectable if the model's gap is ≥ **2.8 pt** (2,513) | needs gap ≥ 28 pt | ✅ choices13k, as long as the model has a gap · ❌ CPC18 |
| **H3C** as written (direction strength vs. human match, *across layers and models*) | not a powered test: n = layers × models, and neighbouring layers are strongly correlated | same | ❌ reformulate (see below) |

**Recommendation for task 3.** Use **choices13k as the main dataset**: all 2,513 ambiguous problems for the mechanistic tests, and the 452 with no-feedback human data for the human comparison. Use **CPC18 as a high-precision replication of the human behavioral comparison only**. Its human rates are very reliable (ceiling 0.96), but 40 games can't support any mechanistic decision rule.

## Full sensitivity table

| Subset | n | Paired MDE d (uncorr. / layer-corr.) | MDE in choice-rate points* | Min model gap for the ⅓ ablation* | AUC CI ± | TOST max readout SD (layer-corr.) | Model–human MDE r (true) | Human ceiling |
|---|---|---|---|---|---|---|---|---|
| c13k, all ambiguous | 2,513 | 0.056 / 0.082 | 0.009 | 0.028 | 0.013 | 0.58 | — | — |
| c13k, two-outcome B | 1,348 | 0.076 / 0.111 | 0.012 | 0.037 | 0.018 | 0.42 | — | — |
| c13k, no-feedback | 452 | 0.132 / 0.193 | 0.022 | 0.065 | 0.032 | 0.25 | 0.19 | 0.71 |
| c13k, no-feedback two-outcome | 254 | 0.176 / 0.259 | 0.028 | 0.085 | 0.042 | 0.18 | 0.26 | 0.69 |
| CPC18, ambiguous | 40 | 0.454 / 0.692 | 0.093 | 0.280 | 0.107 | 0.069 | 0.45 | 0.96 |
| CPC18, two-outcome B | 26 | 0.572 / 0.895 | 0.113 | 0.338 | 0.133 | 0.054 | 0.55 | 0.96 |

\* **Placeholder scale.** The layer-corrected d is converted into choice-rate points using the *human* per-problem SD (0.11–0.14). The model's own SD is unknown until we run a pilot (see next steps). Behavioral tests have no layers, so their uncorrected MDEs are smaller (the first row of the verdict table uses those).

## The human ambiguity effect in our datasets (a side result that matters for tasks 4 and 5)

To have something to compare the MDEs against, I estimated the human ambiguity effect inside each dataset:

1. Train a model on **risky** no-feedback problems to predict the human choice rate from the outcome distributions (out-of-fold R² = 0.71 for choices13k and 0.75 for CPC18).
2. Apply it to the **ambiguous** problems, with B's probabilities set to a mix of the true hidden probabilities (w = 0) and a **uniform prior over B's listed outcomes** (w = 1).
3. Find the w that best explains human behavior. The residual at that w is the ambiguity attitude *beyond* the prior (negative = aversion).

| | choices13k (452) | CPC18 (40) |
|---|---|---|
| Gap vs. a decision maker who knows the true probabilities | **+5.8 pt** [4.4, 7.4] | +1.0 pt [−5.6, 7.3] |
| Gap vs. a uniform prior | +1.1 pt [0.1, 2.1] | **−7.1 pt** [−11.1, −3.0] |
| Best-fitting uniform-prior weight w | **1.0** [0.75, 1.0] | 1.0 [0.25, 1.0] |
| Attitude beyond the prior | **+1.1 pt** [0.2, 2.7], slight seeking | **−7.1 pt** [−10.8, −0.4], aversion |
| Two-outcome-B problems only | −0.7 pt [−2.4, 0.5] (n = 254) | −5.5 pt [−10.6, −0.4] (n = 26) |

**What this means:**
- **choices13k participants behave almost exactly like a uniform-prior decision maker.** That's the H3B pattern, in humans. Their apparent "ambiguity seeking" against the true probabilities is mostly the prior at work: hidden low-probability prizes get treated as 50/50. Beyond the prior, the attitude is about zero.
- **CPC18 participants show real aversion beyond the prior**, of about 7 points. The difference may come from the population (lab vs. MTurk) or from the presentation, and is worth a sentence in the paper.
- **The plan's human target (66% averse in two-urn, 27% in single-urn) is a different quantity:** it's the share of averse *people* in Ellsberg urn tasks, not a choice-rate gap on these problems. Task 4 should consider using these in-dataset estimates instead, or alongside.
- These estimates depend on the baseline model. The CIs bootstrap over problems and don't include uncertainty in the baseline model itself.

## Design recommendations for the pre-registration (task 5)

1. **Build two disclosed twins per ambiguous problem**, not one:
   - **true-probability twin**: B's hidden probabilities, shown in the prompt
   - **uniform twin**: each of B's outcomes shown with probability 1/k

   Then *ambiguous vs. true twin* gives the total gap, and *ambiguous vs. uniform twin* isolates aversion beyond the prior. **H3B then becomes a direct, paired prediction:** the ambiguous activations and choices should match the uniform twin. That's a stronger test than comparing against unmatched risky p = 0.5 trials.
2. **Fix the H3B permutation test.** As written ("no significant separation from p = 0.5 risky trials"), H3B passes whenever the test has no power, and with 1,348 vs. 581 items it will flag even trivial separations. Replace it with an equivalence criterion that has a margin, such as the TOST above or a separation-AUC ≤ 0.6, set in advance.
3. **Reformulate H3C at the item level:** correlate each problem's projection on the direction with the human per-problem gap (452 problems, MDE r = 0.19). Correlating across layers × models has only a handful of non-independent points.
4. **Cross-validate by problem.** Put an ambiguous problem and its twins in the same fold. Otherwise the AUCs leak.
5. **Define "the gap" before ablating.** If the model uses a uniform prior, its gap against the true-probability twin will be large without any aversion. The ⅓-closure rule should be stated against the uniform twin (for aversion) or the true twin (for the total), not left ambiguous.

## Next step: model pilot (replaces the placeholder scale)

Run the chosen model on about 100 choices13k ambiguous problems plus both twins. Read P(choose B) from the logits, and rerun this script with the model's own per-problem SD. It needs the model list (open issue 4 in the plan). The verdict is robust to that: even at 3× the human SD, the choices13k behavioral MDE stays under 2 choice-rate points.
