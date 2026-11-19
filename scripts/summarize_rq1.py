"""Combine the scored RQ1 runs into report tables (CSV + LaTeX) in results/rq1/.

Reads results/generations/<run>/scored.csv for every run (pilot runs starting with "_" skipped).
All rates use eligible (instance, category) pairs only (see ELIGIBLE_ACTIONS in score.py) and
come with 95% Wilson intervals.

Tables:
  main_table    one row per (run, variant): validity, unchanged-file answers, plan-failure rate,
                category agreement, non-trivial failure rate [95% CI], median tokens and seconds,
                and diversity (distinct perturbations among the 3 samples of each prompt)
  category_table  non-trivial failure rate per category, one column per (run, variant)

Usage: python scripts/score.py --run <run> (for each run), then python scripts/summarize_rq1.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from score import wilson

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "rq1"
VARIANTS = ["zero_shot", "few_shot"]
CATEGORIES = ["displaced_object", "blocked_path", "power_loss", "communication_loss", "equipment_fault"]


def load():
    runs = {}
    for d in sorted(GEN.glob("*")):
        if d.is_dir() and not d.name.startswith("_") and (d / "scored.csv").exists():
            runs[d.name] = list(csv.DictReader(open(d / "scored.csv")))
    return runs


def pct(rows, key):
    n, k = len(rows), sum(r[key] == "True" for r in rows)
    lo, hi = wilson(k, n)
    return k, n, 100 * k / n if n else 0.0, lo, hi


def diversity(rows):
    """Mean number of distinct perturbations among the samples of each prompt
    (same instance, category and variant). Answers that changed nothing are not counted."""
    by = defaultdict(set)
    for r in rows:
        if r["valid_perturbation"] == "True" and r["unchanged_file"] != "True":
            by[(r["instance"], r["category"], r["variant"])].add(r["changes"])
    return st.mean(len(v) for v in by.values()) if by else 0.0


def main():
    runs = load()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- main table -------------------------------------------------------------------
    main_rows = []
    for run, rows in runs.items():
        el = [r for r in rows if r["eligible"] == "True"]
        for var in VARIANTS + ["all"]:
            rs = el if var == "all" else [r for r in el if r["variant"] == var]
            row = {"run": run, "variant": var, "n": len(rs)}
            for key in ("valid_perturbation", "unchanged_file", "plan_fails", "category_agree",
                        "nontrivial_failure"):
                k, n, p, lo, hi = pct(rs, key)
                row[key] = round(p, 1)
                if key == "nontrivial_failure":
                    row["nontrivial_k"], row["ci_lo"], row["ci_hi"] = k, round(lo, 1), round(hi, 1)
            row["median_tokens"] = int(st.median(int(r["completion_tokens"]) for r in rs))
            row["median_seconds"] = round(st.median(float(r["seconds"]) for r in rs))
            row["diversity_of_3"] = round(diversity(rs), 2)
            main_rows.append(row)
    with open(OUT / "main_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(main_rows[0]))
        w.writeheader()
        w.writerows(main_rows)

    # ---- category table ---------------------------------------------------------------
    cols = [(run, var) for run in runs for var in VARIANTS]
    cat_rows = []
    for cat in CATEGORIES + ["all"]:
        row = {"category": cat}
        for run, var in cols:
            rs = [r for r in runs[run] if r["eligible"] == "True" and r["variant"] == var
                  and (cat == "all" or r["category"] == cat)]
            k, n, p, _, _ = pct(rs, "nontrivial_failure")
            row[f"{run}|{var}"] = f"{k}/{n}"
        cat_rows.append(row)
    with open(OUT / "category_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0]))
        w.writeheader()
        w.writerows(cat_rows)

    # ---- LaTeX ------------------------------------------------------------------------
    def mode(run):
        return run.split("_", 1)[-1].replace("nothinking", "Thinking off").replace("thinking", "Thinking on")

    def var_name(v):
        return {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "all": "Both"}[v]

    tex = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
           r"Mode & Prompt & Valid & Unchanged & Plan fails & Cat.\ agree & Non-trivial [95\% CI] "
           r"& Tokens & Div. \\", r"\midrule"]
    for r in main_rows:
        tex.append(f"{mode(r['run'])} & {var_name(r['variant'])} & {r['valid_perturbation']:.0f}\\% & "
                   f"{r['unchanged_file']:.0f}\\% & {r['plan_fails']:.0f}\\% & {r['category_agree']:.0f}\\% & "
                   f"{r['nontrivial_failure']:.0f}\\% [{r['ci_lo']:.0f}--{r['ci_hi']:.0f}] & "
                   f"{r['median_tokens']} & {r['diversity_of_3']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "main_table.tex").write_text("\n".join(tex) + "\n")

    def pc(s):
        k, n = map(int, s.split("/"))
        return f"{100 * k / n:.0f}\\% ({k}/{n})"

    tex = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
           "Category & " + " & ".join(f"{mode(run)}, {var_name(v)}" for run, v in cols) + r" \\",
           r"\midrule"]
    for r in cat_rows:
        name = r["category"].replace("_", " ").capitalize() if r["category"] != "all" else "All"
        tex.append(name + " & " + " & ".join(pc(r[f"{run}|{v}"]) for run, v in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "category_table.tex").write_text("\n".join(tex) + "\n")

    # ---- console ----------------------------------------------------------------------
    print(f"{'run':26s} {'variant':10s} {'n':>4s} {'valid':>6s} {'unchg':>6s} {'fails':>6s} {'agree':>6s} "
          f"{'non-trivial [95% CI]':>22s} {'tok':>6s} {'sec':>5s} {'div':>5s}")
    for r in main_rows:
        print(f"{r['run']:26s} {r['variant']:10s} {r['n']:4d} {r['valid_perturbation']:5.0f}% {r['unchanged_file']:5.0f}% "
              f"{r['plan_fails']:5.0f}% {r['category_agree']:5.0f}% "
              f"{r['nontrivial_failure']:7.0f}% [{r['ci_lo']:.0f}-{r['ci_hi']:.0f}]{'':6s} "
              f"{r['median_tokens']:6d} {r['median_seconds']:5d} {r['diversity_of_3']:5.2f}")
    print()
    for r in cat_rows:
        print(f"{r['category']:20s} " + "  ".join(f"{k.split('|')[0][12:]}|{k.split('|')[1]}={v}"
                                                 for k, v in r.items() if k != "category"))
    print(f"\nwrote {OUT}/main_table.{{csv,tex}} and category_table.{{csv,tex}}")


if __name__ == "__main__":
    main()
