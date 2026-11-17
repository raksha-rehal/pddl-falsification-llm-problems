"""RQ1 pipeline steps 2-3: check each LLM-perturbed problem and validate the nominal plan on it.

For every record in results/generations/<run>/raw.jsonl:
  1. Extract the perturbed problem file from the answer (the ```pddl block) and save it.
  2. Valid perturbation? = the file parses, VAL accepts it as well-formed for the domain
     (types/objects consistent), and the goals are unchanged.
  3. Diff it against the nominal problem: which init facts / numeric values changed.
  4. Run VAL with the nominal plan on the perturbed problem. Record whether the plan fails and,
     if so, the first failing step, action and unsatisfied precondition.
  5. Map the unsatisfied precondition to an observed failure category and compare it with the
     requested category. Non-trivial failure = valid + plan fails + small change (1..MAX_CHANGES)
     + no objects deleted + observed category == requested category.
Writes results/generations/<run>/scored.csv and prints summary tables.

Usage: python scripts/score.py --run qwen3.8-27b_thinking
"""
import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pddl_utils as pu

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "rovers-numeric"
DOMAIN = DATA / "domain.pddl"
NOMINAL = ROOT / "results" / "nominal"
VAL = ROOT / "tools" / "VAL" / "build" / "bin" / "Validate"
MAX_CHANGES = 3          # "small" perturbation, matches rule 2 of the prompt

# Which predicates / functions belong to which anomaly category (CLAUDE.md failure table).
CATEGORY_PREDICATES = {
    "displaced_object": {"at_soil_sample", "at_rock_sample"},
    "blocked_path": {"can_traverse"},
    "power_loss": {"energy", "in_sun"},
    "communication_loss": {"visible", "channel_free", "at_lander"},
    "equipment_fault": {"equipped_for_soil_analysis", "equipped_for_rock_analysis",
                        "equipped_for_imaging", "on_board", "calibration_target", "supports",
                        "store_of", "empty"},
}
COMMUNICATE = {"communicate_soil_data", "communicate_rock_data", "communicate_image_data"}


def category_of_predicate(pred: str, action: str = ""):
    """Observed category for an unsatisfied precondition, e.g. 'can_traverse' -> blocked_path.
    A missing (visible ?x ?y) blocks movement when it fails at navigate, so it counts as a
    blocked path there, and as communication loss at a communicate_* action."""
    if pred == "visible":
        return "communication_loss" if action in COMMUNICATE else "blocked_path"
    for cat, preds in CATEGORY_PREDICATES.items():
        if pred in preds:
            return cat
    return "other"


def extract_pddl(content: str):
    """Return (change_sentence, pddl_text) from the model's answer."""
    m = re.search(r"CHANGE:\s*(.+)", content)
    change = m.group(1).strip() if m else ""
    blocks = re.findall(r"```(?:pddl|lisp)?\s*\n(.*?)```", content, re.S | re.I)
    blocks = [b for b in blocks if "(define" in b.lower()]
    if blocks:
        return change, blocks[-1].strip()
    i = content.lower().find("(define")          # no fenced block: take from (define to the end
    return change, (content[i:].strip() if i >= 0 else "")


def run_val(problem: Path, plan: Path) -> dict:
    out = subprocess.run([str(VAL), "-v", str(DOMAIN), str(problem), str(plan)],
                         capture_output=True, text=True).stdout
    r = {"val_wellformed": True, "plan_valid": False, "failure_type": "", "fail_step": "",
         "fail_action": "", "unsat": [], "val_note": ""}
    if "Parser failed" in out or "type-checking" in out or "Type problem" in out:
        r["val_wellformed"] = False
        r["val_note"] = next((l for l in out.splitlines() if "rror" in l or "Type problem" in l), "")
        return r
    if "Plan valid" in out:
        r["plan_valid"] = True
        return r
    m = re.search(r"\((.+?)\) has an unsatisfied precondition at time (\d+)", out)
    if m:
        r.update(failure_type="precondition", fail_action=m.group(1), fail_step=int(m.group(2)))
        advice = out.split("Plan Repair Advice:", 1)[-1]
        # e.g. "(Set (can_traverse rover0 w1 w2) to true)" or "(Satisfy (energy rover3)[=0] >= 5)"
        r["unsat"] = re.findall(r"\((?:Set|Satisfy) \(([^()]+)\)", advice)
    elif "Goal not satisfied" in out:
        r["failure_type"] = "goal_not_satisfied"
        r["unsat"] = re.findall(r"\(Set \(([^()]+)\) to true\)", out.split("Plan Repair Advice:", 1)[-1])
    else:
        r["failure_type"] = "other"
        r["val_note"] = out[-300:]
    return r


def score_record(rec: dict, prob_dir: Path) -> dict:
    row = {k: rec[k] for k in ("job_id", "instance", "category", "variant", "sample", "thinking")}
    usage = rec.get("usage") or {}
    row.update(finish_reason=rec.get("finish_reason", ""), completion_tokens=usage.get("completion_tokens", ""),
               reasoning_chars=len(rec.get("reasoning", "")), seconds=rec.get("seconds", ""))
    change, text = extract_pddl(rec.get("content", ""))
    row["change_text"] = change
    for k in ("extracted", "parses", "val_wellformed", "goals_same", "objects_same", "domain_same",
              "metric_same", "valid_perturbation", "plan_fails", "small_change", "edit_in_category",
              "category_agree", "nontrivial_failure"):
        row[k] = False
    row.update(n_changes="", changes="", failure_type="", fail_step="", fail_action="",
               unsat="", observed_category="", val_note="")
    if not text:
        return row
    row["extracted"] = True
    path = prob_dir / f"{rec['job_id']}.pddl"
    path.write_text(text + "\n")

    nominal = pu.parse_problem((DATA / f"{rec['instance']}.pddl").read_text())
    try:
        pert = pu.parse_problem(text)
    except (ValueError, IndexError) as e:
        row["val_note"] = f"parse: {e}"
        return row
    row["parses"] = True

    d = pu.diff_problems(nominal, pert)
    row.update(n_changes=d["n_changes"], goals_same=d["goals_same"], objects_same=d["objects_same"],
               domain_same=d["domain_same"], metric_same=d["metric_same"])
    row["changes"] = "; ".join(
        [f"-{pu.fmt_fact(f)}" for f in d["removed"]] + [f"+{pu.fmt_fact(f)}" for f in d["added"]] +
        [f"{pu.fmt_fact(k)}: {a:g}->{b:g}" if a is not None and b is not None else f"{pu.fmt_fact(k)}: {a}->{b}"
         for k, a, b in d["fluents_changed"]])
    changed_preds = ({f[0] for f in d["removed"] + d["added"]} |
                     {k[0] for k, _, _ in d["fluents_changed"]})
    row["edit_in_category"] = bool(changed_preds) and changed_preds <= CATEGORY_PREDICATES[rec["category"]]
    row["small_change"] = 1 <= d["n_changes"] <= MAX_CHANGES

    v = run_val(path, NOMINAL / f"{rec['instance']}.plan")
    row.update(val_wellformed=v["val_wellformed"], failure_type=v["failure_type"], fail_step=v["fail_step"],
               fail_action=v["fail_action"], unsat=" | ".join(v["unsat"]), val_note=v["val_note"])
    row["valid_perturbation"] = row["parses"] and v["val_wellformed"] and d["goals_same"]
    row["plan_fails"] = row["valid_perturbation"] and not v["plan_valid"]

    if v["failure_type"] == "precondition" and v["unsat"]:
        action = v["fail_action"].split()[0]
        observed = [category_of_predicate(u.split()[0], action) for u in v["unsat"]]
        row["observed_category"] = observed[0] if len(set(observed)) == 1 else "/".join(sorted(set(observed)))
        row["category_agree"] = rec["category"] in observed
    elif v["failure_type"]:
        row["observed_category"] = v["failure_type"]
    row["nontrivial_failure"] = (row["plan_fails"] and row["small_change"] and d["objects_same"]
                                 and row["category_agree"])
    return row


def rate(rows, key):
    n = len(rows)
    k = sum(1 for r in rows if r[key] is True)
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def summarize(rows):
    keys = ["valid_perturbation", "plan_fails", "small_change", "category_agree", "nontrivial_failure"]
    for group in ("variant", "category"):
        print(f"\n== by {group}")
        print(f"{group:20s} " + " ".join(f"{k:>22s}" for k in keys))
        by = defaultdict(list)
        for r in rows:
            by[r[group]].append(r)
        for g in sorted(by) + ["ALL"]:
            rs = rows if g == "ALL" else by[g]
            print(f"{g:20s} " + " ".join(f"{rate(rs, k):>22s}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "results" / "generations" / args.run
    prob_dir = run_dir / "problems"
    prob_dir.mkdir(exist_ok=True)

    latest = {}                                   # keep the last successful answer per job
    for line in (run_dir / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            latest[r["job_id"]] = r
    rows = [score_record(r, prob_dir) for _, r in sorted(latest.items())]
    with open(run_dir / "scored.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"scored {len(rows)} answers -> {run_dir / 'scored.csv'}")
    summarize(rows)


if __name__ == "__main__":
    main()
