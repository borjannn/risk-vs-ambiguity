"""Task 1: count the ambiguity trials in choices13k and CPC18 from the raw releases.

Writes results/ambiguity_counts.json and prints a summary.
Usage: uv run scripts/download_data.py && uv run scripts/count_ambiguity.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "ambiguity_counts.json"

# Parameters that define a problem (everything except the Amb flag), used to find exact risk twins.
C13K_PARAMS = ["Ha", "pHa", "La", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB", "Corr"]
CPC_PARAMS = ["Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB", "Corr"]


def count_twins(problems, params, amb_col="Amb"):
    """How many ambiguous problems have a risky problem with identical parameters."""
    risky = set(map(tuple, problems.loc[~problems[amb_col], params].values))
    return int(sum(tuple(r) in risky for r in problems.loc[problems[amb_col], params].values))


def choices13k():
    sel = pd.read_csv(RAW / "choices13k" / "c13k_selections.csv")
    gambles = json.loads((RAW / "choices13k" / "c13k_problems.json").read_text())
    # Keys of the JSON are row indices of the selections file.
    sel["n_outcomes_B"] = [len({o for _, o in gambles[str(i)]["B"]}) for i in range(len(sel))]

    amb = sel[sel.Amb]
    amb_nofb = amb[~amb.Feedback]
    problems = sel.drop_duplicates("Problem")
    coverage = amb.groupby("Problem").Feedback.agg(lambda s: "+".join(sorted(map(str, s))))

    return {
        "rows_total": len(sel),
        "problems_total": int(sel.Problem.nunique()),
        "peterson_check_rows_feedback_and_not_amb": int((sel.Feedback & ~sel.Amb).sum()),
        "amb_rows": len(amb),
        "amb_rows_feedback": int(amb.Feedback.sum()),
        "amb_rows_no_feedback": len(amb_nofb),
        "amb_problems": int(amb.Problem.nunique()),
        "amb_problems_no_feedback": int(amb_nofb.Problem.nunique()),
        "amb_problems_by_feedback_coverage": {
            {"True": "feedback_only", "False": "no_feedback_only", "False+True": "both"}[k]: int(v)
            for k, v in coverage.value_counts().items()
        },
        "amb_problems_effective_(B_has_>1_outcome)": int(amb[amb.n_outcomes_B > 1].Problem.nunique()),
        "amb_problems_two_outcome_B": int(amb[amb.n_outcomes_B == 2].Problem.nunique()),
        "amb_problems_two_outcome_B_no_feedback": int(amb_nofb[amb_nofb.n_outcomes_B == 2].Problem.nunique()),
        "amb_participants_per_row": amb.n.describe()[["min", "mean", "max"]].round(1).to_dict(),
        "amb_participant_rows_no_feedback": int(amb_nofb.n.sum()),
        "amb_problems_with_exact_risk_twin": count_twins(problems, C13K_PARAMS),
    }


def cpc18():
    raw = pd.read_csv(RAW / "cpc18" / "all_cpc18_raw_data.csv")
    raw["Amb"] = raw.Amb.astype(bool)
    games = raw.groupby("GameID").agg(
        Set=("Set", "first"), Amb=("Amb", "first"), subjects=("SubjID", "nunique"),
        **{p: (p, "first") for p in CPC_PARAMS},
    )
    assert raw.groupby("GameID").Amb.nunique().max() == 1
    # Feedback is only absent in block 1 (trials 1-5); blocks 2-5 always have full feedback.
    assert (raw.Feedback == (raw.block > 1)).all()

    b_degenerate = (games.LotNumB == 1) & (games.pHb.isin([0, 1]) | (games.Hb == games.Lb))
    two_outcome_b = (games.LotNumB == 1) & games.pHb.between(0, 1, inclusive="neither") & (games.Hb != games.Lb)
    safe_a = (games.LotNumA == 1) & (games.pHa == 1)
    amb_games = games[games.Amb]
    amb_raw = raw[raw.Amb]

    split = pd.cut(games.Set, bins=[0, 5, 7, 9], labels=["CPC15_sets_1-5", "CPC18_exp1_sets_6-7", "CPC18_test_sets_8-9"])
    return {
        "games_total": len(games),
        "subjects_total": int(raw.SubjID.nunique()),
        "choices_total": len(raw),
        "amb_games": int(games.Amb.sum()),
        "games_by_split": {str(k): {"total": len(v), "amb": int(v.Amb.sum())} for k, v in games.groupby(split, observed=True)},
        "calibration_sets_1-7": {"games": int((games.Set <= 7).sum()), "amb": int(games[games.Set <= 7].Amb.sum())},
        "amb_games_effective_(B_has_>1_outcome)": int((games.Amb & ~b_degenerate).sum()),
        "amb_games_two_outcome_B": int((games.Amb & two_outcome_b).sum()),
        "amb_games_with_safe_A": int((games.Amb & safe_a).sum()),
        "amb_subjects_per_game": amb_games.subjects.describe()[["min", "mean", "max"]].round(1).to_dict(),
        "amb_choices_total": len(amb_raw),
        "amb_choices_no_feedback_block": int((~amb_raw.Feedback.astype(bool)).sum()),
        "amb_subject_game_pairs": int(amb_raw.groupby(["SubjID", "GameID"]).ngroups),
        "amb_games_with_exact_risk_twin": count_twins(games, CPC_PARAMS),
    }


def main():
    counts = {"choices13k": choices13k(), "cpc18": cpc18()}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
