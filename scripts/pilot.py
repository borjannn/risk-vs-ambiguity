"""Task 2 follow-up: behavioral model pilot, replacing the placeholder SD in the power analysis.

Runs 100 choices13k ambiguous problems, each in all three conditions, through the hosted
chat models. P(choose B) is read from the first answer token's logprobs. The API has no raw
completion route, so prompts go through the chat template, with "Answer:" moved from the
user message to an assistant prefill (prompt version v2_chat).

The pilot problems come from those with feedback-only human data, so the 452 no-feedback
problems stay untouched for the human comparison. Problems whose true probabilities are
already uniform are skipped, since their `true` and `uniform` prompts are identical. Pilot
problems must be excluded from the confirmatory analyses (see `problem_ids` in the output).

Every stimulus also runs in a swapped version, with the original Option B labelled A and
listed first. Averaging the two versions cancels any bias toward a label or position; the
swap changes both together, so it can't tell those two apart.

Reads FINKI_API_BASE_URL and FINKI_API_KEY from .env. Raw responses are cached in
data/processed/pilot_raw.jsonl, so an interrupted run resumes. Writes results/pilot_results.json.
Usage: uv run scripts/pilot.py [max_problems]
       uv run scripts/pilot.py --analyse-only   (no requests; analyses the cached responses)
"""

import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scipy import stats

from power_analysis import N_LAYERS, mde_paired_d

ROOT = Path(__file__).resolve().parent.parent
STIMULI = ROOT / "data" / "processed" / "stimuli.jsonl"
RAW = ROOT / "data" / "processed" / "pilot_raw.jsonl"
OUT = ROOT / "results" / "pilot_results.json"

MODELS = ("gemma4-26b-a4b-it", "qwen3.8-27b")
PROMPT_VERSION = "v2_chat"
PREFILL = "Answer:"
N_PROBLEMS = 100
N_REPEATS = 10  # stimuli sent a second time, to check that responses are deterministic
N_BOOT = 2000
ALLOWED_HOST = "vllm.finki.ukim.mk"
RETRY_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
MAX_CONSECUTIVE_FAILURES = 5
ALPHA = 0.05
# Remaining problems for each analysis once the pilot problems are excluded.
SIZES = {"human_comparison": 452, "ambiguous_remaining": 2413, "all_twins_remaining": 11922}
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- stimuli

def load_stimuli():
    return [json.loads(line) for line in STIMULI.open(encoding="utf-8")]


def sample_problems(rows, n):
    """Feedback-only ambiguous problems, keeping the population's share of two-outcome Bs."""
    pool = [
        r for r in rows
        if r["dataset"] == "c13k" and r["condition"] == "ambiguous" and r["source_amb"]
        and r["human"]["no_feedback"] is None and not r["uniform_equals_true"]
    ]
    two = sorted(r["problem_id"] for r in pool if r["n_outcomes_B"] == 2)
    many = sorted(r["problem_id"] for r in pool if r["n_outcomes_B"] > 2)
    n_two = round(n * len(two) / len(pool))
    picked = list(rng.choice(two, n_two, replace=False)) + list(rng.choice(many, n - n_two, replace=False))
    return sorted(int(p) for p in picked), len(pool), len(two) / len(pool)


def user_message(prompt, swapped=False):
    assert prompt.endswith("\n" + PREFILL)
    message = prompt.removesuffix("\n" + PREFILL)
    if not swapped:
        return message
    intro, a, b, question = message.split("\n")
    assert a.startswith("Option A: ") and b.startswith("Option B: ")
    return "\n".join([intro, "Option A: " + b.removeprefix("Option B: "), "Option B: " + a.removeprefix("Option A: "), question])


# ---------------------------------------------------------------- API

def read_env():
    env = {}
    for line in (ROOT / ".env").open(encoding="utf-8-sig"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    base, key = env["FINKI_API_BASE_URL"].rstrip("/"), env["FINKI_API_KEY"]
    assert urlparse(base).hostname == ALLOWED_HOST, "refusing to send the key to another host"
    return base, key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # never forward the key to another address


def ask(base, key, model, prompt, swapped=False):
    """One first-token query, with retries. Returns the parsed response, or raises RuntimeError."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_message(prompt, swapped)}, {"role": "assistant", "content": PREFILL}],
        "add_generation_prompt": False,
        "continue_final_message": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    opener = urllib.request.build_opener(NoRedirect)
    error = None
    for attempt in range(MAX_ATTEMPTS):
        req = urllib.request.Request(
            base + "/chat/completions", data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with opener.open(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}".replace(key, "[key]")
            if e.code not in RETRY_CODES:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = type(e).__name__
        time.sleep(5 * 2**attempt)
    raise RuntimeError(error)


def run(rows, problem_ids):
    base, key = read_env()
    done = set()
    if RAW.exists():
        for line in RAW.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["stim_id"], r["repeat"], r.get("swapped", False)))
    picked = set(problem_ids)
    stims = [r for r in rows if r["dataset"] == "c13k" and r["problem_id"] in picked]
    repeat_ids = {r["stim_id"] for r in sorted(stims, key=lambda r: r["stim_id"])[:: max(1, len(stims) // N_REPEATS)][:N_REPEATS]}
    jobs = [(m, s, 0, False) for m in MODELS for s in stims]
    jobs += [(m, s, 1, False) for m in MODELS for s in stims if s["stim_id"] in repeat_ids]
    jobs += [(m, s, 0, True) for m in MODELS for s in stims]
    jobs = [j for j in jobs if (j[0], j[1]["stim_id"], j[2], j[3]) not in done]
    print(f"{len(jobs)} requests to send", flush=True)

    failures = 0
    with RAW.open("a", encoding="utf-8") as f:
        for i, (model, stim, repeat, swapped) in enumerate(jobs, 1):
            try:
                resp = ask(base, key, model, stim["prompt"], swapped)
            except RuntimeError as e:
                failures += 1
                print(f"FAILED {model} {stim['stim_id']}: {e}", flush=True)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    print("too many consecutive failures, stopping; rerun to resume", flush=True)
                    return False
                continue
            failures = 0
            choice = resp["choices"][0]
            items = (choice.get("logprobs") or {}).get("content") or []
            f.write(json.dumps({
                "model": model,
                "stim_id": stim["stim_id"],
                "problem_id": stim["problem_id"],
                "condition": stim["condition"],
                "repeat": repeat,
                "swapped": swapped,
                "prompt_version": PROMPT_VERSION,
                "text": (choice.get("message") or {}).get("content"),
                "top_logprobs": [[t["token"], t["logprob"]] for t in items[0].get("top_logprobs", [])] if items else [],
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
            }) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    return failures == 0


# ---------------------------------------------------------------- analysis

def read_answer(top):
    """P(B | A or B), its logit, the A + B mass, and whether one option fell outside the top 20."""
    lp = {}
    for token, logprob in top:
        k = token.strip()
        if k in ("A", "B"):
            lp[k] = np.logaddexp(lp[k], logprob) if k in lp else logprob
    if not lp:
        return None
    floor = min(l for _, l in top)
    a, b = lp.get("A", floor), lp.get("B", floor)
    pa, pb = math.exp(a), math.exp(b)
    return {"p_b": pb / (pa + pb), "logit_b": b - a, "mass": pa + pb, "censored": len(lp) < 2}


def summarise_gap(x):
    boot = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return {
        "mean": round(float(x.mean()), 4),
        "ci95": [round(float(v), 4) for v in np.percentile(x[boot].mean(axis=1), [2.5, 97.5])],
        "sd": round(float(x.std(ddof=1)), 4),
        "sd_ci95": [round(float(v), 4) for v in np.percentile(x[boot].std(axis=1, ddof=1), [2.5, 97.5])],
    }


def power_from_sd(sd):
    """What the power analysis's tests can detect if this is the model's per-problem gap SD."""
    out = {}
    for name, n in SIZES.items():
        d, d_layer = mde_paired_d(n, ALPHA), mde_paired_d(n, ALPHA / N_LAYERS)
        out[name] = {
            "n": n,
            "behavioral_mde": round(d * sd, 4),
            # Smallest gap at which a shift of gap/3 is detectable (the rule the original script sized).
            "min_gap_ablation_detect_any_closure": round(3 * d_layer * sd, 4),
            # Smallest gap at which the lower bound clears 1/3 if the true closure is 1/2.
            "min_gap_ablation_lower_bound_third": round(6 * d_layer * sd, 4),
        }
    return out


def analyse_swap(records, stim_by_id):
    """Compares the two orderings. All P and logit values are for the ORIGINAL Option B."""
    conds = ("ambiguous", "true", "uniform")
    versions = {False: {}, True: {}}
    for r in records:
        ans = read_answer(r["top_logprobs"]) if r["repeat"] == 0 else None
        if ans:
            swapped = r.get("swapped", False)
            versions[swapped][(r["problem_id"], r["condition"])] = (1 - ans["p_b"], -ans["logit_b"]) if swapped else (ans["p_b"], ans["logit_b"])
    ids = sorted(p for p in {k[0] for k in versions[True]} if all((p, c) in versions[s] for c in conds for s in versions))
    if not ids:
        return {"problems_complete": 0}
    get = lambda s, c, i: np.array([versions[s][(p, c)][i] for p in ids])
    p_o, p_s = {c: get(False, c, 0) for c in conds}, {c: get(True, c, 0) for c in conds}
    l_o, l_s = {c: get(False, c, 1) for c in conds}, {c: get(True, c, 1) for c in conds}
    p_avg = {c: (p_o[c] + p_s[c]) / 2 for c in conds}
    l_avg = {c: (l_o[c] + l_s[c]) / 2 for c in conds}
    contrasts = (("ambiguous_minus_true", "ambiguous", "true"), ("ambiguous_minus_uniform", "ambiguous", "uniform"), ("true_minus_uniform", "true", "uniform"))
    d_true = np.array([stim_by_id[f"c13k-{i}-true"]["ev_B_true"] - stim_by_id[f"c13k-{i}-true"]["ev_A"] for i in ids])
    d_uni = np.array([stim_by_id[f"c13k-{i}-true"]["ev_B_uniform"] - stim_by_id[f"c13k-{i}-true"]["ev_A"] for i in ids])
    sure_a = np.array([len(stim_by_id[f"c13k-{i}-true"]["A"]) == 1 for i in ids])
    gap = p_avg["ambiguous"] - p_avg["true"]
    spearman = lambda a, b: round(float(stats.spearmanr(a, b).statistic), 3)
    return {
        "problems_complete": len(ids),
        "mean_p_original_b": {c: {"original_order": round(float(p_o[c].mean()), 4), "swapped_order": round(float(p_s[c].mean()), 4), "averaged": round(float(p_avg[c].mean()), 4)} for c in conds},
        # Above 0.5 means the model favors the label B, the second-listed option, or both.
        "mean_p_label_b": {c: round(float(((p_o[c] + 1 - p_s[c]) / 2).mean()), 4) for c in conds},
        "bias_to_label_b_logit": {c: summarise_gap((l_o[c] - l_s[c]) / 2) for c in conds},
        "gaps_probability_original_order": {n: summarise_gap(p_o[a] - p_o[b]) for n, a, b in contrasts},
        "gaps_probability_swapped_order": {n: summarise_gap(p_s[a] - p_s[b]) for n, a, b in contrasts},
        "gaps_probability_averaged": {n: summarise_gap(p_avg[a] - p_avg[b]) for n, a, b in contrasts},
        "gaps_logit_averaged": {n: summarise_gap(l_avg[a] - l_avg[b]) for n, a, b in contrasts},
        "gap_averaged_ambiguous_minus_true_by_a_type": {"a_is_sure_thing": summarise_gap(gap[sure_a]), "a_is_a_gamble": summarise_gap(gap[~sure_a])},
        "power_if_sd_equals_averaged_ambiguous_minus_true": power_from_sd(float(gap.std(ddof=1))),
        "power_if_sd_equals_averaged_ambiguous_minus_uniform": power_from_sd(float((p_avg["ambiguous"] - p_avg["uniform"]).std(ddof=1))),
        "sanity_averaged": {
            "share_true_choosing_higher_ev": round(float(((l_avg["true"] > 0) == (d_true > 0)).mean()), 3),
            "share_uniform_choosing_higher_stated_ev": round(float(((l_avg["uniform"] > 0) == (d_uni > 0)).mean()), 3),
            "spearman_true_logit_vs_true_ev_diff": spearman(l_avg["true"], d_true),
            "spearman_uniform_logit_vs_uniform_ev_diff": spearman(l_avg["uniform"], d_uni),
            "spearman_ambiguous_logit_with_true": spearman(l_avg["ambiguous"], l_avg["true"]),
            "spearman_ambiguous_logit_with_uniform": spearman(l_avg["ambiguous"], l_avg["uniform"]),
        },
    }


def analyse_model(model, records, stim_by_id):
    first = [r for r in records if r["repeat"] == 0 and not r.get("swapped", False)]
    table = {}
    for r in first:
        ans = read_answer(r["top_logprobs"])
        table[(r["problem_id"], r["condition"])] = {**(ans or {}), "readable": ans is not None, "tokens": r["prompt_tokens"]}
    conds = ("ambiguous", "true", "uniform")
    ids = sorted(p for p in {k[0] for k in table} if all(table.get((p, c), {}).get("readable") for c in conds))
    get = lambda c, f: np.array([table[(p, c)][f] for p in ids])
    p, l = {c: get(c, "p_b") for c in conds}, {c: get(c, "logit_b") for c in conds}
    mass = np.concatenate([get(c, "mass") for c in conds])
    censored = np.concatenate([get(c, "censored") for c in conds])
    all_p, all_l = np.concatenate(list(p.values())), np.concatenate(list(l.values()))

    def ev_diff(c):
        key = {"true": "ev_B_true", "uniform": "ev_B_uniform"}[c]
        return np.array([stim_by_id[f"c13k-{i}-true"][key] - stim_by_id[f"c13k-{i}-true"]["ev_A"] for i in ids])

    d_true, d_uni = ev_diff("true"), ev_diff("uniform")
    spearman = lambda a, b: round(float(stats.spearmanr(a, b).statistic), 3)

    repeats = {r["stim_id"]: r for r in records if r["repeat"] == 1}
    originals = {r["stim_id"]: r for r in first}
    deltas = []
    for sid, r in repeats.items():
        a, b = read_answer(r["top_logprobs"]), read_answer(originals[sid]["top_logprobs"]) if sid in originals else None
        if a and b:
            deltas.append(abs(a["logit_b"] - b["logit_b"]))

    gaps = {
        "ambiguous_minus_true": p["ambiguous"] - p["true"],
        "ambiguous_minus_uniform": p["ambiguous"] - p["uniform"],
        "true_minus_uniform": p["true"] - p["uniform"],
    }
    logit_gaps = {
        "ambiguous_minus_true": l["ambiguous"] - l["true"],
        "ambiguous_minus_uniform": l["ambiguous"] - l["uniform"],
        "true_minus_uniform": l["true"] - l["uniform"],
    }
    two = np.array([stim_by_id[f"c13k-{i}-true"]["n_outcomes_B"] == 2 for i in ids])
    sure_a = np.array([len(stim_by_id[f"c13k-{i}-true"]["A"]) == 1 for i in ids])
    return {
        "label_swap": analyse_swap(records, stim_by_id),
        "mean_p_b_true_by_ev": {
            "b_has_higher_ev": round(float(p["true"][d_true > 0].mean()), 4),
            "b_has_lower_ev": round(float(p["true"][d_true < 0].mean()), 4),
        },
        "mean_p_b_ambiguous_by_uniform_ev": {
            "b_has_higher_ev": round(float(p["ambiguous"][d_uni > 0].mean()), 4),
            "b_has_lower_ev": round(float(p["ambiguous"][d_uni < 0].mean()), 4),
        },
        "gap_ambiguous_minus_true_by_a_type": {
            "a_is_sure_thing": summarise_gap(gaps["ambiguous_minus_true"][sure_a]),
            "a_is_a_gamble": summarise_gap(gaps["ambiguous_minus_true"][~sure_a]),
        },
        "problems_complete": len(ids),
        "stimuli_unreadable": sum(not v["readable"] for v in table.values()),
        "answer_mass": {"mean": round(float(mass.mean()), 4), "min": round(float(mass.min()), 4), "share_below_0.99": round(float((mass < 0.99).mean()), 3)},
        "stimuli_with_one_option_outside_top20": int(censored.sum()),
        "mean_p_b": {c: round(float(p[c].mean()), 4) for c in conds},
        "share_saturated": {
            "p_below_.01_or_above_.99": round(float(((all_p < 0.01) | (all_p > 0.99)).mean()), 3),
            "p_below_.05_or_above_.95": round(float(((all_p < 0.05) | (all_p > 0.95)).mean()), 3),
        },
        "share_answering_B": {c: round(float((p[c] > 0.5).mean()), 3) for c in conds},
        "gaps_probability": {k: summarise_gap(v) for k, v in gaps.items()},
        "gaps_logit": {k: summarise_gap(v) for k, v in logit_gaps.items()},
        "gaps_probability_two_outcome": {k: summarise_gap(v[two]) for k, v in gaps.items()},
        "n_two_outcome": int(two.sum()),
        "power_if_sd_equals_ambiguous_minus_true": power_from_sd(float(gaps["ambiguous_minus_true"].std(ddof=1))),
        "power_if_sd_equals_ambiguous_minus_uniform": power_from_sd(float(gaps["ambiguous_minus_uniform"].std(ddof=1))),
        "sanity": {
            "spearman_true_logit_vs_true_ev_diff": spearman(l["true"], d_true),
            "share_true_choosing_higher_ev": round(float(((p["true"] > 0.5) == (d_true > 0)).mean()), 3),
            "spearman_uniform_logit_vs_uniform_ev_diff": spearman(l["uniform"], d_uni),
            "spearman_ambiguous_logit_vs_true_ev_diff": spearman(l["ambiguous"], d_true),
            "spearman_ambiguous_logit_vs_uniform_ev_diff": spearman(l["ambiguous"], d_uni),
        },
        "ambiguous_closeness": {
            "mean_abs_p_diff_to_true": round(float(np.abs(gaps["ambiguous_minus_true"]).mean()), 4),
            "mean_abs_p_diff_to_uniform": round(float(np.abs(gaps["ambiguous_minus_uniform"]).mean()), 4),
            "spearman_logit_with_true": spearman(l["ambiguous"], l["true"]),
            "spearman_logit_with_uniform": spearman(l["ambiguous"], l["uniform"]),
        },
        "prompt_tokens_mean": {c: round(float(get(c, "tokens").mean()), 1) for c in conds},
        "prompt_token_diff_ambiguous_minus_true": {
            "mean": round(float((get("ambiguous", "tokens") - get("true", "tokens")).mean()), 2),
            "sd": round(float((get("ambiguous", "tokens") - get("true", "tokens")).std(ddof=1)), 2),
        },
        # Share of A-minus-B logits lying on a multiple of 1/8, which would point to coarse logits.
        "share_logit_gaps_on_eighths": round(float((np.abs(all_l * 8 - np.round(all_l * 8)) < 0.02).mean()), 3),
        "distinct_logit_values": int(len(np.unique(np.round(all_l, 3)))),
        "determinism": {"repeats": len(deltas), "max_abs_logit_change": round(max(deltas), 4) if deltas else None},
    }


def main():
    rows = load_stimuli()
    args = sys.argv[1:]
    analyse_only = "--analyse-only" in args
    n = int(args[0]) if args and not analyse_only else N_PROBLEMS
    problem_ids, pool_size, two_share = sample_problems(rows, N_PROBLEMS)
    finished = None if analyse_only else run(rows, problem_ids[:n])

    records = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    stim_by_id = {r["stim_id"]: r for r in rows}
    out = {
        "last_run_finished": finished,
        "prompt_version": PROMPT_VERSION,
        "prefill": PREFILL,
        "pool": {"feedback_only_ambiguous_non_uniform": pool_size, "two_outcome_share": round(two_share, 3)},
        "problem_ids": problem_ids,
        "layers_for_correction": N_LAYERS,
        "models": {m: analyse_model(m, [r for r in records if r["model"] == m], stim_by_id) for m in MODELS},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["models"], indent=2))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
