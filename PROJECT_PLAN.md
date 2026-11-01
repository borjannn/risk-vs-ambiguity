# Risk vs. Ambiguity Dissociation in LLMs

**Dissociating Risk from Ambiguity in Large Language Models: Evidence for a Genuine Ambiguity-Aversion Mechanism versus a Uniform-Prior Default**

This file keeps only the parts of the original combined plan that apply to this project; the RL mode-collapse and personality projects have been dropped.
The task list with owners is in [`TASKS.md`](TASKS.md), and the reading order is in [`READING_LIST.md`](READING_LIST.md).

---

## 1. The question

Behavioral economics treats **risk** (the probabilities are known) and **ambiguity** (the probabilities are unknown or vague) as two different things. The Ellsberg paradox shows that people prefer a known 50/50 bet over an unknown bet with the same expected value (EV).

choices13k and CPC18 both contain ambiguity trials, but almost every paper that uses LLMs as cognitive models tests risky choice only. Peterson et al. (2021) even remove the ambiguous trials in their own preprocessing, which is the clearest sign that the gap is real.

**What we ask:** does the model treat ambiguity as its own category, as humans do (H3A)? Or does it quietly turn ambiguity into risk by assuming a uniform prior and treating the missing probability as noise (H3B)?

**Positioning:** Zhu, Yan & Griffiths (2025, arXiv:2505.11615) already extract risk steering vectors in this domain, and report that their behavioral-to-neural alignment method beats contrastive extraction. So our diff-of-means pipeline is the **baseline**, not the state of the art. We run both.

**Main risk to the project:** it only works if the ambiguity subset is large enough. Task 1 counted it: **2,513** ambiguous problems in choices13k (452 with no-feedback human data) and **40** of 270 games in CPC18 (see [`results/task01_ambiguity_counts.md`](results/task01_ambiguity_counts.md)). The **power analysis is still the go/no-go gate**.

---

## 2. Hypotheses and decision rules

These thresholds must be frozen in the pre-registration **before the first causal test runs** (task 5 in [`TASKS.md`](TASKS.md)).

| ID | Hypothesis | Test and decision rule | If it fails |
|---|---|---|---|
| **H3A** | The model uses a separate direction for ambiguous choices. | Diff-of-means between risk and ambiguity trials matched on EV, then ablate and measure the shift toward the pure-EV prediction. **Rule:** held-out AUC ≥ 0.75 **AND** ablation closes at least ⅓ of the measured ambiguity gap. | No separate direction survives EV matching: move to H3B. |
| **H3B** | The model does not represent ambiguity. It assumes a uniform prior and treats the missing probability as noise. This is the more interesting outcome, written as a positive prediction so it can be tested directly. | Train a probe to read stated probability on risky trials, then read its output on ambiguous trials. **Rule:** mean readout within 0.05 of 0.5, **and** no significant separation from the matched p = 0.5 risky trials in a permutation test. | The readout is spread out or consistently away from 0.5, and the ambiguous activations sit outside the p = 0.5 cluster: this points back to H3A. |
| **H3C** | The strength of the ambiguity direction predicts how closely the model matches the size of the human ambiguity-aversion effect. | Correlate direction strength with the distance from the human target, across layers and models. **Rule:** choose the target first: about 66% ambiguity-averse in two-urn designs, about 27% in single-urn designs. H3C cannot be tested until one design is chosen and justified. | No relationship: the direction exists but does not control how strong the aversion is. |
| **H3D** | The direction responds to unknown probability, not to uncertain language in general. | Build the probe set before extraction, with hedged-but-stated probabilities ("about 40%") as distractors. **Rule:** the mean projection is at least **2×** larger for unknown-probability items than for hedged-but-stated items. | It responds equally to both: it is a hedging or vagueness feature. Still a real finding, but a different paper. |

How they fit together: H3A and H3B are **competing** explanations. H3C and H3D only apply if a direction exists, and H3D is what turns "we found a direction" into "we found an *ambiguity* direction".

> Note: the general plan also uses the IDs H3A–H3E for the personality project. In this repo, H3x always means the hypotheses above.

---

## 3. Methods

### Core methods (we run all of these)

| Family | Method | What it shows and how we run it | Pitfalls and required controls |
|---|---|---|---|
| Experimental design | **Confound matching** (EV, outcome range, token length) | That the contrast isolates ambiguity. Use propensity or exact matching, and publish a balance table. | **The most likely failure:** a direction that looks like ambiguity but is really EV or length. |
| Statistics | **Power analysis before extraction** | Whether the data can answer the question at all. This is our scoping gate. | An underpowered null on ambiguity tells us nothing. Do it during scoping, not as an appendix. |
| Direction extraction | **Diff-of-means** (contrastive activation) | A candidate linear direction separating risk and ambiguity trials. Run it at every layer and report the whole sweep. | The contrast set *is* the experiment. Picking the best layer afterwards and reporting only that overstates the result. |
| Direction extraction | **Supervised linear probe** (logistic / ridge) | Whether the information can be read out linearly. An independent second read on the same contrast, and the basis of the H3B probability-readout probe. | Being readable is not the same as being used. Run a **shuffled-label control task** (Hewitt & Liang). |
| Direction extraction | **Behavioral-to-neural alignment** (self-alignment, Lasso; Zhu, Yan & Griffiths) | A direction aligned to a behavioral measure (MCMC or certainty equivalents) instead of to prompt wording. Our main alternative pipeline. | It is already reported to beat contrastive extraction in this exact domain. Running both is what separates a baseline result from a current one. |
| Direction extraction | **Low-rank subspace search** | Whether a rank-k subspace explains the behaviour better than a single direction. Run a rank sweep before claiming one direction. | If rank 3 clearly beats rank 1, the main claim changes. Work on refusal has moved from "one direction" to "a cone or subspace", so reviewers will ask. |
| Causal intervention | **Directional ablation** (projection removal) | Necessity. Sweep the layers and the ablation weight, and report the whole curve. | Ablating at one layer and at all layers gives different answers. Always add a capability check. |
| Causal intervention | **Activation steering with a coefficient sweep** | Sufficiency and graded control. Report the whole dose-response curve. | Fragile out of distribution and sensitive to prompt format. Report in-distribution and out-of-distribution results **separately**. |
| Validation | **Specificity probe set** (synthetic Ellsberg set with hedged distractors) | That the direction encodes ambiguity rather than a surface cue (H3D). | Build it **before** extracting the direction. A probe set written afterwards tests nothing. |
| Validation | **Within-method ceiling and across-construct floor** | Makes the cosine between the diff-of-means and alignment directions meaningful. Run two independent extractions per method, plus one unrelated concept. | Without these two anchors, "cos = 0.5" means nothing. |
| Control | **Capability-retention checks** (MMLU subset, held-out perplexity, format compliance) | That the ablation is surgical. | Report them even when they show no effect. |
| Statistics | **Bootstrap CIs, permutation tests, FDR across layers** | Calibrated uncertainty for every headline number. | Correct across layers, and also show the uncorrected sweep. The power statement belongs in the Results. |
| Robustness | **Second model family** (and the second dataset if possible) | That the result is not a quirk of one checkpoint. | Plan for it from the start, because replications added at the end tend not to happen. |
| Methodology | **Pre-registration of decision rules** | That our null results are credible. Freeze the hypothesis table and timestamp it (an OSF upload or a tagged git commit). | "A null result is informative" only holds if the thresholds were set before the causal tests. |

### Fallbacks and optional methods (only if needed)

- **Distributed alignment search**: use it if the behaviour is clearly causal but diff-of-means finds nothing (the mechanism may be a rotated subspace).
- **SAE feature attribution** (Gemma Scope, if Gemma-2 is one of our models): gives named features. Use SAEs to *interpret* and diff-of-means to *control*; don't mix the two roles.
- **Activation patching**: gives a layer × position heatmap of where the ambiguity signal enters. Fix the metric and corruption method in advance.
- **Inference-time intervention** (attention-head steering): an alternative injection site if residual-stream steering hurts fluency.

### Out of scope (belongs to the other projects)

Crosscoder model diffing, checkpoint trajectory tracking, path/attribution patching, ACDC, the cross-construct interference matrix and the external text-scorer battery (LIWC etc.).

---

## 4. Data

| Dataset | Role | Notes |
|---|---|---|
| **CPC18 / CPC15** ([zenodo](https://zenodo.org/records/2571510)) | **Replication** of the behavioral and human comparison (decided in task 3) | 270 games (210 calibration + 60 test), 694,500 choices, 926 subjects; **40 ambiguous games** (28 in calibration, 12 in test). (The original plan's "210 problems, 240 participants" counts only the calibration set and only Experiment 1's subjects.) The **only** dataset with a *designed* ambiguity condition (risk / ambiguity / decisions from experience). The ambiguity cell is small, so the power analysis comes first. Keep the **first no-feedback block separate from the feedback blocks**, or learning will confound the contrast. CPC15 is a 150-problem subset of CPC18, so it is **not** an independent test set. |
| **choices13k** ([github](https://github.com/jcpeterson/choices13k)) | **Main dataset** (decided in task 3) | More than 13k two-gamble problems, 14,711 participants and over 1M choices, released as aggregate choice rates. It has ambiguous-option trials, but they are incidental, and Peterson et al. drop them (along with the no-feedback trials), leaving 9,831 risky tasks. **The size of the ambiguity subset is not published anywhere, so we count it ourselves.** |
| **Psych-101 / Centaur** ([github](https://github.com/marcelbinz/Llama-3.1-Centaur-70B)) | Comparison subject model | 160 experiments, 60,092 participants, 10.68M choices. It contains choices13k-family gamble tasks, so **check contamination** (compare its task list against our evaluation problems) before any claim about generalisation. |
| **Synthetic Ellsberg probe set** (to build) | **Supporting**: H3D specificity probe (decided in task 3) | 200–500 items: two-urn and single-urn framings × gain and loss framings, plus hedged-but-stated distractors ("about 40%"). It is not validated on humans, so it is a specificity probe rather than the human anchor. It is also a contribution on its own: document how it was built. |

**Reference model for "pure-EV prediction":** humans do not use EV, so pick and state a reference model (EV, CPT or BEAST) before running H3A.

---

## 5. Who does what

The split is by workstream, so each person owns a coherent piece of the pipeline end to end.

**Borjan: data and direction/causal pipeline**
Count the ambiguity trials · split risk and ambiguity trials · EV matching and balance table · run models and cache activations · diff-of-means · behavioral-to-neural alignment and direction comparison · ablation · steering · capability check · second-model replication · code release · proofreading.
Writing: Introduction, Related Work (LLM cognitive modeling; steering and extraction), Methods.

**Andrej: behavior, human anchor and probes**
Power analysis · human ambiguity-aversion target · synthetic Ellsberg probe set · behavioral baseline and human comparison · Centaur baseline and contamination check · linear probe with control task · H3B probability readout · p = 0.5 permutation test · specificity (H3D) · H3C correlation · loss-framing robustness · statistics.
Writing: Related Work (Ellsberg and behavioral economics; human effect sizes), Limitations.

**Both:** choosing the main dataset · pre-registration · Results · Discussion · slides.

### Critical path

```
Count ambiguity trials (B) ─► Power analysis (A) ─► Choose dataset (both) + Human target (A) ─► Pre-registration (both, FROZEN)
                                                                                                        │
Split trials + EV matching (B) ─┐                                                                       ▼
Synthetic Ellsberg set (A) ─────┼─► Cache activations (B) ─► extraction / probes / behavior ─► ablation, steering, H3D (only after the freeze)
```

- The first four steps are the gate. **Nothing else is worth starting in earnest until the power analysis says go.** In the meantime the reading list and the Related Work drafts can run in parallel.
- The Ellsberg probe set must exist **before** any direction is extracted.
- Draft both readings of the Discussion (dissociation vs. uniform prior) **before** looking at the ablation result.

---

## 6. Open issues found while distilling the plan

1. **Wrong attribution on the effect-size link.** `10.1007/s11238-016-9544-1` is **Voorhoeve, Binmore, Stefansson & Stewart (2016), "Ambiguity attitudes, framing, and consistency"**, *Theory and Decision* 81:313–337. It is not by Trautmann & van de Kuilen (the general plan also misspells that name as "Van der Kuijlen").
2. **That paper partly contradicts our framing assumption.** Its abstract reports that, *contrary to most other studies*, switching from a gain to a loss frame does **not** remove ambiguity aversion in their single-urn design. The loss-framing task assumes that aversion "weakens or reverses for losses", so cite both sides and don't state the reversal as settled.
3. **The 66% / 27% numbers are not yet traced to a source.** The plan credits them to "Chew, Ratchford & Sagi, as summarised in…". Find the actual table before the human-target task locks one in.
4. **No model list.** The general plan mentions "the model list" (with Gemma-2-9B on it), but neither file contains one. We still need to choose: a primary model, a second family, and whether running Centaur (70B) is feasible on our compute.
5. **Missing links** in the general plan have been filled in `READING_LIST.md` where they could be checked.
