"""Task 2: power (sensitivity) analysis for every test in the pre-registration.

The number of problems is fixed by the datasets, so for each analysis and each subset we
compute the minimum detectable effect (MDE) at alpha = .05 and 80% power, both uncorrected
and Bonferroni-corrected across layers.

To read the MDEs against something real, we also estimate the HUMAN ambiguity effect
inside each dataset. A gradient-boosted model is trained on risky (Amb = 0) no-feedback
problems to predict the human choice rate from the outcome distributions. It is then
applied to the ambiguous problems with B's probabilities set to a mix of the true hidden
probabilities (w = 0) and a uniform prior over B's listed outcomes (w = 1). The w that
best explains human behavior is the human "uniform-prior weight". The mean residual
at that w is the ambiguity attitude beyond the prior (negative = aversion to B).

Writes results/power_analysis.json.
Usage: uv run scripts/power_analysis.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from gambles import load_c13k, load_cpc18, load_cpc18_raw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "power_analysis.json"

ALPHA, POWER = 0.05, 0.80
N_LAYERS = 42  # Gemma-2-9B, the deepest candidate model: the most conservative layer correction
TOST_MARGIN = 0.05  # H3B: mean probe readout within 0.05 of 0.5
AUC_TARGET = 0.75  # H3A
W_GRID = np.linspace(0, 1, 21)
N_BOOT = 2000
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- minimum detectable effects

def mde_paired_d(n, alpha):
    """Smallest Cohen's d a two-sided paired / one-sample t-test detects with POWER."""
    df, t_crit = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)
    # The lower tail is negligible but returns NaN for large noncentrality, so treat NaN as 0.
    power = lambda d: stats.nct.sf(t_crit, df, d * np.sqrt(n)) + np.nan_to_num(stats.nct.cdf(-t_crit, df, d * np.sqrt(n)))
    d_normal = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(POWER)) / np.sqrt(n)
    return optimize.brentq(lambda d: power(d) - POWER, d_normal / 2, d_normal * 3)


def mde_correlation(n, alpha):
    """Smallest Pearson r detectable (Fisher z approximation)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(POWER)
    return np.tanh(z / np.sqrt(n - 3))


def auc_ci_halfwidth(n_per_class, auc=AUC_TARGET):
    """95% CI half-width of a held-out AUC (Hanley & McNeil, 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    n1 = n2 = n_per_class
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2)
    return 1.96 * np.sqrt(var)


def tost_max_sd(n, alpha=ALPHA, margin=TOST_MARGIN):
    """Largest SD of the probe readout at which TOST still shows |mean - 0.5| < margin
    with POWER, if the true mean is exactly 0.5."""
    t = stats.t.ppf(1 - alpha, n - 1) + stats.t.ppf(1 - (1 - POWER) / 2, n - 1)
    return margin * np.sqrt(n) / t


# ---------------------------------------------------------------- human ambiguity effect

def features(dist_A, dist_B, corr):
    f = [corr]
    for d in (dist_A, dist_B):
        o, p = np.array(list(d)), np.array(list(d.values()))
        ev = (o * p).sum()
        f += [ev, np.sqrt((p * (o - ev) ** 2).sum()), o.min(), o.max(), p[o < 0].sum()]
    return f + [f[6] - f[1]]  # EV_B - EV_A


def with_prior(dist_B, w):
    """B's distribution if the decision maker mixes the true probabilities with a uniform prior."""
    k = len(dist_B)
    return {o: (1 - w) * p + w / k for o, p in dist_B.items()}


def human_gap(df):
    """df: one row per problem with dist_A, dist_B, Corr, Amb, rate, rate_se2 (no-feedback)."""
    risky, amb = df[~df.Amb], df[df.Amb]
    X = np.array([features(r.dist_A, r.dist_B, r.Corr) for r in risky.itertuples()])
    model = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10, random_state=0)
    oof = cross_val_predict(model(), X, risky.rate, cv=KFold(10, shuffle=True, random_state=0))
    fit = model().fit(X, risky.rate)

    # preds[w_index, problem]: predicted human rate if B's probabilities were the w-mix.
    preds = np.array([
        fit.predict(np.array([features(r.dist_A, with_prior(r.dist_B, w), r.Corr) for r in amb.itertuples()]))
        for w in W_GRID
    ])
    y = amb.rate.to_numpy()
    two_outcome = np.array([len(d) == 2 for d in amb.dist_B])

    def summarise(mask):
        res = y[mask] - preds[:, mask]  # residuals for every w
        boot = rng.integers(0, mask.sum(), size=(N_BOOT, mask.sum()))

        def best(r):  # w that best explains behavior (allowing a constant attitude shift)
            i = r.var(axis=-1).argmin(axis=0)
            return i, r.mean(axis=-1)

        i, means = best(res)
        bres = res[:, boot]  # (w, boot, problem)
        bi, bmeans = best(bres)
        b_delta = bmeans[bi, np.arange(N_BOOT)]
        ci = lambda a: [round(float(x), 3) for x in np.percentile(a, [2.5, 97.5])]
        resid_best = res[i]
        return {
            "n_problems": int(mask.sum()),
            "gap_vs_true_probabilities": {"mean": round(float(means[0]), 3), "ci95": ci(bmeans[0]), "sd": round(float(res[0].std()), 3)},
            "gap_vs_uniform_prior": {"mean": round(float(means[-1]), 3), "ci95": ci(bmeans[-1]), "sd": round(float(res[-1].std()), 3)},
            "best_uniform_prior_weight": {"w": round(float(W_GRID[i]), 2), "ci95": ci(W_GRID[bi])},
            "attitude_beyond_prior": {"mean": round(float(means[i]), 3), "ci95": ci(b_delta), "sd": round(float(resid_best.std()), 3)},
            # Share of between-problem variance in the human residual that is not sampling noise.
            "reliability_of_human_gap": round(float(1 - amb.rate_se2.to_numpy()[mask].mean() / resid_best.var()), 2),
        }

    return {
        "baseline_r2_on_risky_out_of_fold": round(float(1 - np.var(risky.rate - oof) / np.var(risky.rate)), 2),
        "baseline_residual_sd_on_risky": round(float(np.std(risky.rate - oof)), 3),
        "all_ambiguous": summarise(np.ones(len(amb), bool)),
        "two_outcome_B": summarise(two_outcome),
    }


# ---------------------------------------------------------------- main

def main():
    c13k = load_c13k()
    raw = load_cpc18_raw()
    cpc = load_cpc18(raw)

    # The CPC18 reconstruction must reproduce every payoff observed in the raw data.
    support = {g: (set(cpc.dist_A[g]), set(cpc.dist_B[g])) for g in cpc.index}
    outside = sum((~s.Apay.isin(support[g][0])).sum() + (~s.Bpay.isin(support[g][1])).sum() for g, s in raw.groupby("GameID"))
    assert outside == 0, outside

    c13k_nofb = c13k[~c13k.Feedback]
    gaps = {"choices13k_no_feedback": human_gap(c13k_nofb), "cpc18_block1": human_gap(cpc)}

    two = lambda df: df.dist_B.map(len) == 2
    c_amb = c13k[c13k.Amb].drop_duplicates("Problem")
    subsets = {
        # name: (n problems, human no-feedback data?, human-gap summary that sets the effect-size scale)
        # Model-only subsets borrow the human SD of their dataset as the effect-size scale.
        "c13k_all_ambiguous": (len(c_amb), False, ("choices13k_no_feedback", "all_ambiguous")),
        "c13k_two_outcome": (int(two(c_amb).sum()), False, ("choices13k_no_feedback", "two_outcome_B")),
        "c13k_no_feedback": (int(c13k_nofb.Amb.sum()), True, ("choices13k_no_feedback", "all_ambiguous")),
        "c13k_no_feedback_two_outcome": (int((c13k_nofb.Amb & two(c13k_nofb)).sum()), True, ("choices13k_no_feedback", "two_outcome_B")),
        "cpc18_ambiguous": (int(cpc.Amb.sum()), True, ("cpc18_block1", "all_ambiguous")),
        "cpc18_two_outcome": (int((cpc.Amb & two(cpc)).sum()), True, ("cpc18_block1", "two_outcome_B")),
    }

    sensitivity = {}
    for name, (n, human, ref) in subsets.items():
        a_corr = ALPHA / N_LAYERS
        row = {
            "n": n,
            "has_human_no_feedback_data": human,
            "paired_mde_d": round(mde_paired_d(n, ALPHA), 3),
            "paired_mde_d_layer_corrected": round(mde_paired_d(n, a_corr), 3),
            "auc_ci95_halfwidth_at_0.75": round(auc_ci_halfwidth(n), 3),
            "tost_max_readout_sd": round(tost_max_sd(n), 3),
            "tost_max_readout_sd_layer_corrected": round(tost_max_sd(n, a_corr), 3),
        }
        g = gaps[ref[0]][ref[1]]
        sd = g["attitude_beyond_prior"]["sd"]
        row |= {
            # Converting d into choice-rate points, using the human residual SD as the scale.
            # Placeholder until the model pilot gives the model's own SD.
            "paired_mde_choice_rate_points_if_sd_as_humans": round(mde_paired_d(n, a_corr) * sd, 3),
            "min_model_gap_for_ablation_third_detectable": round(3 * mde_paired_d(n, a_corr) * sd, 3),
        }
        if human:
            rel = max(g["reliability_of_human_gap"], 1e-6)
            row |= {
                "model_human_mde_r_observed": round(mde_correlation(n, ALPHA), 3),
                # The model's own gap is noise-free (read from logits), so only human noise attenuates r.
                "model_human_mde_r_true": round(min(mde_correlation(n, ALPHA) / np.sqrt(rel), 1.0), 3),
                "human_correlation_ceiling": round(float(np.sqrt(rel)), 2),
                "human_attitude_beyond_prior": g["attitude_beyond_prior"]["mean"],
            }
        sensitivity[name] = row

    p_half = lambda df: (~df.Amb & two(df) & df.dist_B.map(lambda d: 0.5 in d.values())).sum()
    out = {
        "settings": {"alpha": ALPHA, "power": POWER, "layers_for_correction": N_LAYERS, "tost_margin": TOST_MARGIN, "auc_target": AUC_TARGET},
        "human_ambiguity_effect": gaps,
        "sensitivity": sensitivity,
        "h3b_comparison_set_risky_two_outcome_p_half": {
            "c13k_unique_problems": int(p_half(c13k.drop_duplicates("Problem"))),
            "c13k_no_feedback": int(p_half(c13k_nofb)),
            "cpc18": int(p_half(cpc)),
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
