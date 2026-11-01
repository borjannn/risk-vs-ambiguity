# Task 1: Ambiguity trials in choices13k and CPC18

Counted directly from the raw releases. To reproduce:

```
uv run scripts/download_data.py
uv run scripts/count_ambiguity.py   # writes results/ambiguity_counts.json
```

In both datasets, ambiguity is the `Amb` flag on **Option B**: B's outcomes are shown, but its probabilities are not. Option A is always fully described.

## Headline numbers

| | choices13k | CPC18 (full raw release) |
|---|---|---|
| Problems / games in total | 13,006 | **270** (210 calibration + 60 test) |
| **Ambiguous problems** | **2,513** (19%) | **40** (15%) |
| … with a no-feedback (description-only) human condition | **452** | 40 (block 1 of every game) |
| … where B has exactly 2 outcomes (where H3B's 0.5 prediction applies cleanly) | 1,348 (254 of them no-feedback) | 26 |
| … whose B is effectively ambiguous (B has more than one outcome, so hiding probabilities hides something) | 2,513 (all) | 40 (all) |
| … with an exact risk twin (same parameters, `Amb = 0`) | 10 | 0 |
| Human data per ambiguous problem | ~17 participants (15–33), aggregated `bRate` only | 80–125 subjects, trial-level |
| Human no-feedback observations on ambiguous problems | 7,326 participant-rows (×5 trials each) | 20,890 choices (4,178 subject-game pairs × 5 trials) |

CPC18 ambiguous games by split: CPC15 sets 1–5: **19** of 150 · CPC18 Experiment 1, sets 6–7: **9** of 60 · CPC18 test, sets 8–9: **12** of 60. Calibration-only (sets 1–7): **28** of 210. In 17 of the 40, Option A is a sure thing.

choices13k ambiguous problems by feedback condition: 2,061 feedback only · 296 both · 156 no-feedback only.

## Checks against the plan

- ✅ **Peterson et al.'s 9,831 risky tasks** is exactly the number of choices13k rows with feedback and without ambiguity. Their filter was "drop ambiguous trials and no-feedback trials", as the plan says.
- ❌ **"CPC18 has only 210 choice problems in total"**: the raw release has **270** games, and 210 is the calibration set only. The 694,500 choices include the 60 test games.
- ❌ **"240 participants"**: that's CPC18 Experiment 1 (sets 6–7) only. The full release has **926** subjects.
- The dataset dictionary PDF also describes only sets 1–7, but the CSV contains sets 1–9.

## What this means for the next tasks

1. **The bottleneck is human comparison items, not stimuli.** The direction-extraction and probe analyses (H3A, H3B, H3D) only need problems to show the model, so all **2,513** choices13k ambiguous problems are usable. Only the analyses that compare against humans (the behavioral baseline and H3C) need a no-feedback human rate. That limits them to **452** choices13k problems or **40** CPC18 games. The power analysis (task 2) should treat these two cases separately.
2. **Use no-feedback human rates only.** With feedback, participants see outcomes and gradually learn B's hidden distribution, so ambiguity turns into experience. In choices13k, feedback and no-feedback blocks were also not run sequentially (see its README).
3. **Neither dataset gives us risk twins, but we can build perfect ones.** Only 10 of 2,513 choices13k problems have an exact risky twin, and CPC18 has none, so matching on EV from the existing risky problems can only be approximate. However, the releases include B's *true* hidden probabilities. So for every ambiguous problem we can create its risky twin by **revealing those probabilities in the prompt**: same outcomes, same EV, the only difference being the disclosure. That's the cleanest possible contrast for diff-of-means. It has no human rate, but the mechanistic contrast doesn't need one. Worth deciding in task 3 and building in task 13.
4. **H3B's "≈ 0.5" readout is only well defined when B has two outcomes**, because a uniform prior over k outcomes gives 1/k each. Restrict that test to the 1,348 two-outcome problems (254 with no-feedback human data), or state the k-outcome prediction in the pre-registration.
5. **CPC18 is small but precise**: 40 games with about 104 subjects each. It's better suited as a high-precision human anchor and a replication set than as the main source of stimuli. The 12 test-set games are the only ambiguous items with a held-out split from the competition.
