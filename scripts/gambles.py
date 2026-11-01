"""Load choices13k and CPC18 problems as explicit outcome distributions.

Each problem gets `dist_A` and `dist_B`: dicts {outcome: probability} holding the TRUE
probabilities. For ambiguous problems (Amb = 1) participants saw B's outcomes but not
these probabilities.
"""

import json
from math import comb
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _merge(pairs):
    out = {}
    for outcome, p in pairs:
        if p > 0:
            out[outcome] = out.get(outcome, 0.0) + p
    return out


def cpc_dist(H, pH, L, lot_shape, lot_num):
    """CPC15/18 distribution of an option (Erev et al., 2017, Appendix A).

    With probability pH the payoff is drawn from a lottery with expected value H,
    otherwise it is L. The lottery shape is '-' (single outcome), 'Symm' (binomial
    around H), 'R-skew' or 'L-skew' (geometric probabilities 1/2, 1/4, ...).
    """
    if lot_shape == "-":
        lottery = [(H, 1.0)]
    elif lot_shape == "Symm":
        k = lot_num - 1
        lottery = [(H - k / 2 + i, comb(k, i) * 0.5**k) for i in range(lot_num)]
    elif lot_shape in ("R-skew", "L-skew"):
        sign, C = (1, -lot_num - 1) if lot_shape == "R-skew" else (-1, lot_num + 1)
        lottery = [(H + C + sign * 2 ** (i + 1), 2.0 ** -(i + 1)) for i in range(lot_num)]
        lottery[-1] = (lottery[-1][0], lottery[-1][1] * 2)
    else:
        raise ValueError(lot_shape)
    return _merge([(o, p * pH) for o, p in lottery] + [(L, 1 - pH)])


def load_c13k():
    """One row per choices13k selections row (problem x feedback condition)."""
    sel = pd.read_csv(RAW / "choices13k" / "c13k_selections.csv")
    gambles = json.loads((RAW / "choices13k" / "c13k_problems.json").read_text())
    # Keys of the JSON are row indices of the selections file.
    sel["dist_A"] = [_merge((o, p) for p, o in gambles[str(i)]["A"]) for i in range(len(sel))]
    sel["dist_B"] = [_merge((o, p) for p, o in gambles[str(i)]["B"]) for i in range(len(sel))]
    # Human uncertainty of the aggregate rate: bRate_std is the SD of per-subject P(B) across subjects.
    sel["rate"] = sel.bRate
    sel["rate_se2"] = sel.bRate_std**2 / sel.n
    return sel


def load_cpc18_raw():
    raw = pd.read_csv(RAW / "cpc18" / "all_cpc18_raw_data.csv")
    raw["Amb"] = raw.Amb.astype(bool)
    return raw


def load_cpc18(raw=None):
    """One row per CPC18 game, with the human no-feedback (block 1) choice rate."""
    raw = load_cpc18_raw() if raw is None else raw
    games = raw.groupby("GameID").first()[
        ["Set", "Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB", "Amb", "Corr"]
    ]
    games["dist_A"] = [cpc_dist(r.Ha, r.pHa, r.La, r.LotShapeA, r.LotNumA) for r in games.itertuples()]
    games["dist_B"] = [cpc_dist(r.Hb, r.pHb, r.Lb, r.LotShapeB, r.LotNumB) for r in games.itertuples()]
    # Block 1 (trials 1-5) is the only block without feedback.
    per_subject = raw[raw.block == 1].groupby(["GameID", "SubjID"]).B.mean()
    by_game = per_subject.groupby("GameID")
    games["n"] = by_game.size()
    games["rate"] = by_game.mean()
    games["rate_se2"] = by_game.var() / by_game.size()
    return games
