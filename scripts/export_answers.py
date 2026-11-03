"""Exports every pilot question with both models' answers to one spreadsheet.

One row per question: the 100 choices13k problems in three versions and two orders (600 rows)
and the four Ellsberg prompts. The tested option is the one whose description changes: the
original Option B in choices13k, and the unknown urn in Ellsberg. Repeat requests are left out.

Writes results/pilot_answers.csv (UTF-8 with BOM, so Excel reads it correctly).
Usage: uv run scripts/export_answers.py
"""

import csv
import json

from pilot import MODELS, RAW, ROOT, load_stimuli, read_answer, user_message

OUT = ROOT / "results" / "pilot_answers.csv"
ELLSBERG = ROOT / "results" / "ellsberg_demo.json"


def main():
    stims = {r["stim_id"]: r for r in load_stimuli()}
    rows = {}

    def row(key, **fields):
        return rows.setdefault(key, fields)

    for line in RAW.open(encoding="utf-8"):
        r = json.loads(line)
        if r["repeat"] != 0:
            continue
        stim, swapped = stims[r["stim_id"]], r.get("swapped", False)
        twin = stims[f"c13k-{r['problem_id']}-true"]
        ans = read_answer(r["top_logprobs"])
        out = row(
            (r["problem_id"], r["condition"], swapped),
            question_set="choices13k",
            problem_id=r["problem_id"],
            version=r["condition"],
            order="swapped" if swapped else "original",
            tested_option_label="A" if swapped else "B",
            ev_A=round(twin["ev_A"], 2),
            ev_B_true=round(twin["ev_B_true"], 2),
            ev_B_uniform=round(twin["ev_B_uniform"], 2),
            question=user_message(stim["prompt"], swapped),
        )
        answer = r["text"].strip()
        out[f"{r['model']}_answer"] = answer
        out[f"{r['model']}_chose_tested_option"] = answer == out["tested_option_label"]
        out[f"{r['model']}_p_tested_option"] = round(1 - ans["p_b"] if swapped else ans["p_b"], 4)

    for r in json.loads(ELLSBERG.read_text())["rows"]:
        tested = "A" if r["known_urn_label"] == "B" else "B"
        out = row(
            ("ellsberg", r["win_color"], r["known_urn_label"]),
            question_set="ellsberg",
            problem_id="",
            version=f"win_on_{r['win_color']}",
            order="swapped" if tested == "A" else "original",
            tested_option_label=tested,
            ev_A="",
            ev_B_true="",
            ev_B_uniform="",
            question=r["user_message"],
        )
        out[f"{r['model']}_answer"] = r["answer"]
        out[f"{r['model']}_chose_tested_option"] = r["answer"] == tested
        out[f"{r['model']}_p_tested_option"] = round(1 - r["p_known_urn"], 4)

    table = sorted(rows.values(), key=lambda r: (r["question_set"], str(r["problem_id"]).zfill(6), r["version"], r["order"]))
    columns = list(table[0])[:10] + [f"{m}_{f}" for m in MODELS for f in ("answer", "chose_tested_option", "p_tested_option")]
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(table)
    print(f"{len(table)} questions written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
