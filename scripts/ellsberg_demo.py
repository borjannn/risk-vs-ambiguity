"""The classic two-urn Ellsberg question, asked directly through the hosted chat models.

Each model chooses between an urn with a known 50/50 mix and an urn with an unknown mix,
betting on red and then on black, with each urn in each position (2 x 2 = 4 prompts per
model). Choosing the known urn for BOTH colors is the Ellsberg pattern: no single belief
about the unknown urn can justify it. The answer is read the same way as in pilot.py.

This is a small demonstration, not the task 13 probe set.
Writes results/ellsberg_demo.json.
Usage: uv run scripts/ellsberg_demo.py
"""

import json

from pilot import MODELS, PREFILL, ROOT, ask, read_answer, read_env, user_message

OUT = ROOT / "results" / "ellsberg_demo.json"

KNOWN = "an urn with 50 red balls and 50 black balls"
UNKNOWN = "an urn with 100 red and black balls in an unknown ratio"
PROMPT = (
    "You are choosing between two options. Each option draws one ball at random from an urn.\n"
    "Option A: {known}. You win 100 if the ball is {color}, otherwise you win nothing.\n"
    "Option B: {unknown}. You win 100 if the ball is {color}, otherwise you win nothing.\n"
    "Which option do you choose? Answer with A or B.\n"
    + PREFILL
)


def main():
    base, key = read_env()
    rows = []
    for model in MODELS:
        for color in ("red", "black"):
            for swapped in (False, True):
                prompt = PROMPT.format(known=KNOWN, unknown=UNKNOWN, color=color)
                resp = ask(base, key, model, prompt, swapped)
                choice = resp["choices"][0]
                items = choice["logprobs"]["content"]
                top = [[t["token"], t["logprob"]] for t in items[0]["top_logprobs"]]
                ans = read_answer(top)
                # Option B is the unknown urn in the original order and the known urn when swapped.
                p_known = ans["p_b"] if swapped else 1 - ans["p_b"]
                rows.append({
                    "model": model,
                    "win_color": color,
                    "known_urn_label": "B" if swapped else "A",
                    "user_message": user_message(prompt, swapped),
                    "answer": choice["message"]["content"].strip(),
                    "chose_known_urn": choice["message"]["content"].strip() == ("B" if swapped else "A"),
                    "p_known_urn": round(p_known, 4),
                    "answer_mass": round(ans["mass"], 4),
                })
                print(f"{model} | win on {color} | known urn is {rows[-1]['known_urn_label']} | "
                      f"answer {rows[-1]['answer']} | P(known urn) = {p_known:.3f} | mass {ans['mass']:.3f}", flush=True)

    summary = {}
    for model in MODELS:
        m = [r for r in rows if r["model"] == model]
        by_color = {c: sum(r["p_known_urn"] for r in m if r["win_color"] == c) / 2 for c in ("red", "black")}
        summary[model] = {
            "times_chose_known_urn": sum(r["chose_known_urn"] for r in m),
            "out_of": len(m),
            "mean_p_known_urn": round(sum(r["p_known_urn"] for r in m) / len(m), 4),
            "mean_p_known_urn_by_win_color": {c: round(v, 4) for c, v in by_color.items()},
            "ellsberg_pattern": all(v > 0.5 for v in by_color.values()),
        }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
