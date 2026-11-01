# Tasks

Owners and the reasoning behind the split are in [`PROJECT_PLAN.md`](PROJECT_PLAN.md#5-who-does-what). Update the **Status** column as you go (`To Do` → `In Progress` → `Done`) and put results or links in **Notes / Output**.

## Phase 1: Scoping (the go/no-go gate)

| # | Status | Task | Assigned | Link | Notes / Output |
|---|---|---|---|---|---|
| 1 | Done | Count the ambiguity trials in choices13k and CPC18 directly from the raw releases | Borjan | [link](https://zenodo.org/records/2571510) | Do this first. The source paper's preprocessing drops ambiguous-option trials, so the published numbers do not answer this. **Output:** choices13k 2,513 ambiguous problems (452 no-feedback); CPC18 40 of 270 games. See [`results/task01_ambiguity_counts.md`](results/task01_ambiguity_counts.md). |
| 2 | Done | Run the power analysis: the smallest ambiguity effect we could detect at each subset size | Andrej |  | The go/no-go gate for the project. An underpowered null on ambiguity would tell us nothing, which defeats the design. **Output:** GO with choices13k as main (2,513 mechanistic / 452 human-comparison problems); CPC18 only as a behavioral replication. Humans in choices13k behave like a uniform prior (H3B-like); CPC18 humans show ~7 pt aversion beyond it. Rerun with the model's own SD after a pilot. See [`results/task02_power_analysis.md`](results/task02_power_analysis.md). |
| 3 | Done | Choose the main dataset, and decide whether the synthetic Ellsberg set is the main or the supporting stimulus set | Borjan & Andrej |  | CPC18 has a designed ambiguity condition but only 40 ambiguous games (of 270; task 1). choices13k is larger, but its ambiguity trials are incidental. **Output:** choices13k is the main dataset; CPC18 replicates the behavioral and human comparison; the synthetic Ellsberg set is supporting (H3D). Twins built for every problem. See [`results/task03_dataset_and_twins.md`](results/task03_dataset_and_twins.md). |
| 4 | To Do | Fix the human ambiguity-aversion target and name the design it comes from | Andrej | [link](https://link.springer.com/article/10.1007/s11238-016-9544-1) | About 66% averse in two-urn designs, and about 27% in single-urn designs. H3C cannot be tested until one is chosen and justified. |
| 5 | To Do | Write the pre-registration, including the H3B positive prediction and all thresholds | Borjan & Andrej |  | H3B is the interesting outcome, so its prediction has to be fixed in advance, or it will read as a null explained away afterwards. |

## Phase 2: Introduction and Related Work

| # | Status | Task | Assigned | Link | Notes / Output |
|---|---|---|---|---|---|
| 6 | To Do | Write the Introduction: risk and ambiguity as separate psychological categories, the Ellsberg paradox, and why most LLM cognitive-modeling papers test risky choice only | Borjan |  | The gap is the contribution. Peterson et al. removing ambiguity trials in their own preprocessing is the clearest evidence that the gap is real. |
| 7 | To Do | Related Work: the Ellsberg paradox and the risk/ambiguity distinction in behavioral economics | Andrej |  | From Ellsberg (1961) through the modern surveys of ambiguity attitudes. Include prospect theory for the choice of reference model. |
| 8 | To Do | Related Work: papers on LLM rationality and cognitive modeling (for example Centaur) that test risky choice only | Borjan | [link](https://www.nature.com/articles/s41586-025-09215-4) | Centaur and the shortcut critique, plus the RL/SFT paper for the reinforcement learning version. |
| 9 | To Do | Related Work: the size of ambiguity-aversion effects in humans, and how they depend on design and framing | Andrej |  | Saying clearly that the human effect size depends on the design answers the obvious objection to H3C. |
| 10 | To Do | Related Work: steering and direction extraction, including the behavioral-to-neural alignment result for risk preference | Borjan | [link](https://arxiv.org/abs/2505.11615) | The closest prior work, and it beats contrastive extraction in exactly this domain. Present our pipeline as the baseline. |

## Phase 3: Data and stimuli

| # | Status | Task | Assigned | Link | Notes / Output |
|---|---|---|---|---|---|
| 11 | Done | Separate risk-only trials from ambiguity trials in the chosen dataset, and document the criterion | Borjan |  | In CPC18, keep the no-feedback block separate from the feedback blocks, or learning will confound the contrast. **Output:** Criterion: the dataset's `Amb` flag; human rates attached only to the condition humans saw, with no-feedback and feedback rates kept separate (`scripts/build_stimuli.py`). |
| 12 | In Progress | Match risk and ambiguity trials on EV, outcome range and token length | Borjan |  | Without EV matching, any direction we find is a value feature rather than an ambiguity feature. Publish the balance table. **Output:** EV and outcome range matched exactly by the twin construction; ambiguous prompts ~10 characters longer. Token counts pending the model's tokenizer. Balance in [`results/stimuli_summary.json`](results/stimuli_summary.json). |
| 13 | To Do | Build the synthetic Ellsberg probe set: two-urn and single-urn framings, gain and loss framings, plus distractors with hedged but stated probabilities | Andrej |  | Build it before extraction. The hedged distractors are what make H3D falsifiable. |

## Phase 4: Experiments

| # | Status | Task | Assigned | Link | Notes / Output |
|---|---|---|---|---|---|
| 14 | To Do | Run every trial through each model and cache the activations at the decision position | Borjan |  |  |
| 15 | To Do | Behavioral baseline: test whether the model shows any ambiguity aversion, meaning a shift in choice rate toward the known option beyond what EV predicts | Andrej |  | Worth reporting on its own, because many papers skip ambiguity entirely. |
| 16 | To Do | Compare the model's aversion against the human target chosen above | Andrej |  |  |
| 17 | To Do | Repeat the behavioral baseline on the Centaur adapter, after a contamination check | Andrej |  | If a model fitted to human data shows aversion that the base model lacks, the claim gets much stronger. Compare Psych-101's task list against our evaluation problems first. |
| 18 | To Do | Extract the candidate ambiguity direction with diff-of-means between risk and ambiguity trials, matched on EV, at every layer | Borjan |  |  |
| 19 | To Do | Train a linear probe with a shuffled-label control task, as an independent read on the same contrast | Andrej |  |  |
| 20 | To Do | Implement behavioral-to-neural alignment as a second extraction method, and compare the two directions | Borjan |  | Running both is what separates a baseline result from a current one. |
| 21 | To Do | Test H3A against H3B: train a probe to read stated probability on risky trials, then read its output on ambiguous trials | Andrej |  | This is what turns H3B from an absence into a positive prediction we can test. Expect about 0.5 if the model assumes a uniform prior. |
| 22 | To Do | Test whether activations on ambiguous trials sit inside the distribution of risky trials at p = 0.5 | Andrej |  | Use a permutation test against the matched risky trials. |
| 23 | To Do | Causal test: ablate the candidate ambiguity direction and check whether the choice rate on ambiguous trials moves toward the pure-EV prediction | Borjan |  | The decisive test. Sweep layers and ablation weights, and report the curve. |
| 24 | To Do | Steer in both directions to test sufficiency and graded control | Borjan |  | Report in-distribution and out-of-distribution effects separately: steering vectors are fragile out of distribution. |
| 25 | To Do | Capability check after ablation | Borjan |  |  |
| 26 | To Do | Specificity check: test whether the direction also responds to other kinds of uncertainty, such as 'maybe' or 'around X%', which a real ambiguity detector should not | Andrej |  | H3D. If it responds equally to hedged-but-stated items, the direction is a hedging feature and the claim of the paper has to change. |
| 27 | To Do | Correlate the strength of the ambiguity direction with how closely the model's ambiguous-trial choice rate matches the human effect size | Andrej |  | H3C. |
| 28 | To Do | Framing robustness: repeat the headline analysis with loss framing | Andrej |  | Human ambiguity aversion weakens or reverses for losses, so a direction that ignores framing tells us something. |
| 29 | To Do | Replicate on a second model family and, if possible, on the other dataset | Borjan |  |  |

## Phase 5: Statistics, writing and release

| # | Status | Task | Assigned | Link | Notes / Output |
|---|---|---|---|---|---|
| 30 | To Do | Statistics: effect sizes, confidence intervals, permutation tests, FDR across layers, and a clear statement of power | Andrej |  | Given the size of the subsets, the power statement belongs in the Results, not buried in an appendix. |
| 31 | To Do | Write the Methods: dataset split, matching, both extraction pipelines, and the ablation protocol | Borjan |  |  |
| 32 | To Do | Write the Results, with figures for the behavioral baseline, the direction comparison, the H3B probe readout, the ablation and the specificity test | Borjan & Andrej |  |  |
| 33 | To Do | Write the Discussion: is the dissociation supported, or does the model turn ambiguity into risk with an assumed prior? | Borjan & Andrej |  | Draft both readings before looking at the ablation result. |
| 34 | To Do | Write the Limitations: subset size and power, dataset composition, steering reliability, and framing sensitivity | Andrej |  |  |
| 35 | To Do | Release the code, the synthetic probe set and the analysis notebooks | Borjan |  | The probe set is a useful contribution on its own, so document how it was built. |
| 36 | To Do | Format and proofread the final paper | Borjan |  |  |
| 37 | To Do | Prepare the final presentation slides | Borjan & Andrej |  |  |
