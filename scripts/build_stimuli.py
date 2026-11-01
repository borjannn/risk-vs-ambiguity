"""Task 3: build the stimulus set with ambiguity twins.

Every problem in choices13k and CPC18 is rendered in up to three conditions that differ
ONLY in how Option B's probabilities are described:

    ambiguous  B's outcomes listed, probabilities hidden ("with probability unknown")
    true       B's true probabilities shown (the hidden ones, for ambiguous problems)
    uniform    each of B's k outcomes shown with probability 1/k

Option A is always fully described. Problems whose B has a single outcome only get the
`true` condition, since hiding the probability of a sure thing hides nothing.
`human_condition` marks the row showing what participants actually saw (ambiguous for
Amb = 1 problems, true otherwise); only those rows carry human choice rates.

Writes data/processed/stimuli.jsonl (gitignored, rebuild with this script) and
results/stimuli_summary.json.
Usage: uv run scripts/build_stimuli.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gambles import load_c13k, load_cpc18

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "stimuli.jsonl"
SUMMARY = ROOT / "results" / "stimuli_summary.json"

PROMPT_VERSION = "v1"
PROMPT = (
    "You are choosing between two options.\n"
    "Option A: {A}\n"
    "Option B: {B}\n"
    "Which option do you choose? Answer with A or B.\n"
    "Answer:"
)
CONDITIONS = ("ambiguous", "true", "uniform")


def fmt_outcome(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def fmt_prob(p):
    return f"{float(f'{p * 100:.4g}'):g}%"


def describe(dist, hidden=False):
    """Outcomes in descending order, so all conditions of a problem list them identically."""
    items = sorted(dist.items(), key=lambda kv: -kv[0])
    if len(items) == 1 and not hidden:
        return f"{fmt_outcome(items[0][0])} for sure"
    if hidden:
        return ", ".join(f"{fmt_outcome(o)} with probability unknown" for o, _ in items)
    return ", ".join(f"{fmt_outcome(o)} with probability {fmt_prob(p)}" for o, p in items)


def ev(dist):
    return float(sum(o * p for o, p in dist.items()))


def pairs(dist):
    return [[float(o), float(p)] for o, p in sorted(dist.items(), key=lambda kv: -kv[0])]


def stimuli_for(dataset, problem_id, dist_A, dist_B, amb, extra, human):
    k = len(dist_B)
    uniform = {o: 1 / k for o in dist_B}
    shown = {"ambiguous": None, "true": dist_B, "uniform": uniform}
    conditions = CONDITIONS if k >= 2 else ("true",)
    seen_by_humans = "ambiguous" if amb else "true"
    for cond in conditions:
        B = describe(dist_B, hidden=True) if cond == "ambiguous" else describe(shown[cond])
        prompt = PROMPT.format(A=describe(dist_A), B=B)
        yield {
            "stim_id": f"{dataset}-{problem_id}-{cond}",
            "dataset": dataset,
            "problem_id": int(problem_id),
            "condition": cond,
            "source_amb": bool(amb),
            "human_condition": cond == seen_by_humans,
            "n_outcomes_B": k,
            "uniform_equals_true": bool(all(abs(p - 1 / k) < 1e-9 for p in dist_B.values())),
            "A": pairs(dist_A),
            "B_true": pairs(dist_B),
            "B_shown": None if shown[cond] is None else pairs(shown[cond]),
            "ev_A": ev(dist_A),
            "ev_B_true": ev(dist_B),
            "ev_B_uniform": ev(uniform),
            "ev_B_shown": None if shown[cond] is None else ev(shown[cond]),
            **extra,
            "human": human if cond == seen_by_humans else None,
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
            "prompt_chars": len(prompt),
        }


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def c13k_stimuli():
    sel = load_c13k()
    for pid, rows in sel.groupby("Problem"):
        nofb, fb = rows[~rows.Feedback], rows[rows.Feedback]
        human = {
            "no_feedback": None if nofb.empty else {"rate_B": float(nofb.rate.iloc[0]), "n": int(nofb.n.iloc[0]), "sd_B": float(nofb.bRate_std.iloc[0])},
            "feedback": None if fb.empty else {"rate_B": float(fb.rate.iloc[0]), "n": int(fb.n.iloc[0]), "sd_B": float(fb.bRate_std.iloc[0]), "block": int(fb.Block.iloc[0])},
        }
        r = rows.iloc[0]
        yield from stimuli_for("c13k", pid, r.dist_A, r.dist_B, r.Amb, {"corr": int(r.Corr)}, human)


def cpc18_stimuli():
    games = load_cpc18()
    split = lambda s: "cpc15" if s <= 5 else "cpc18_exp1" if s <= 7 else "cpc18_test"
    for gid, g in games.iterrows():
        human = {
            "no_feedback": {"rate_B": float(g.rate), "n": int(g.n), "se2": nan_to_none(float(g.rate_se2))},
            "feedback_blocks_2_5": {"rate_B": float(g.rate_feedback)},
        }
        extra = {"corr": int(g.Corr), "cpc_set": int(g.Set), "cpc_split": split(g.Set)}
        yield from stimuli_for("cpc18", gid, g.dist_A, g.dist_B, g.Amb, extra, human)


def summarise(df):
    out = {"prompt_version": PROMPT_VERSION, "prompt_template": PROMPT, "rows": len(df)}
    for ds, d in df.groupby("dataset"):
        triples = d.groupby("problem_id").condition.transform("size") == 3
        t = d[triples]
        out[ds] = {
            "problems": int(d.problem_id.nunique()),
            "problems_with_all_three_conditions": int(t.problem_id.nunique()),
            "problems_true_only_(single-outcome_B)": int(d[~triples].problem_id.nunique()),
            "source_ambiguous_problems": int(d[d.source_amb].problem_id.nunique()),
            "source_ambiguous_with_no_feedback_human_rate": int(d[d.human_condition & d.source_amb].human.map(lambda h: h["no_feedback"] is not None).sum()),
            "uniform_twin_identical_to_true_twin": int(t[t.condition == "uniform"].uniform_equals_true.sum()),
            # Balance across the three conditions of the same problems.
            "balance": {
                cond: {
                    "prompt_chars_mean": round(float(g.prompt_chars.mean()), 1),
                    "ev_B_shown_minus_true_mean_abs": None if cond == "ambiguous" else round(float((g.ev_B_shown - g.ev_B_true).abs().mean()), 3),
                }
                for cond, g in t.groupby("condition")
            },
        }
    return out


def main():
    rows = list(c13k_stimuli()) + list(cpc18_stimuli())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarise(pd.DataFrame(rows))
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\n{len(rows)} stimuli written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
