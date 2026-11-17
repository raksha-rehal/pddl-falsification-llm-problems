"""Pipeline step 0: nominal plans for the IPC-3 Rovers numeric problems.

In plain terms, for each Rovers problem file this script:
  1. asks the planner (Metric-FF) to solve it, which gives the "nominal plan",
  2. saves that plan to a file,
  3. asks VAL to double-check that the plan really works on that problem,
  4. counts some basic facts about the problem (rovers, waypoints, goals, ...),
and finally writes everything into one table: results/nominal/summary.csv.

Later steps reuse these nominal plans: we run each one against the LLM-perturbed
problems and see where it breaks.

Usage: python scripts/make_nominal.py [first] [last]   (default 1 10)
"""
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

# --- Where everything lives -------------------------------------------------
# ROOT is the repo folder (this file is in ROOT/scripts/, so go up two levels).
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "rovers-numeric"            # domain + instance-N.pddl files
DOMAIN = DATA / "domain.pddl"                      # the Rovers domain (actions, predicates)
FF = ROOT / "tools" / "Metric-FF-v2.1" / "ff"      # the planner program
VAL = ROOT / "tools" / "VAL" / "build" / "bin" / "Validate"  # the plan checker program
OUT = ROOT / "results" / "nominal"                 # where plans, logs and the CSV go
TIMEOUT_S = 300                                    # give the planner at most 5 minutes per problem


def run_ff(problem: Path):
    """Run Metric-FF on one problem file.

    Returns (plan, seconds, status):
      plan    - list of actions like "(navigate rover0 waypoint3 waypoint1)", or None
      seconds - how long the planner took
      status  - "ok", "no plan" or "timeout"
    """
    t0 = time.time()
    try:
        # Equivalent to typing:  ff -o domain.pddl -f instance-N.pddl
        # We capture what the planner prints so we can read the plan out of it.
        out = subprocess.run([str(FF), "-o", str(DOMAIN), "-f", str(problem)],
                             capture_output=True, text=True, timeout=TIMEOUT_S).stdout
    except subprocess.TimeoutExpired:
        # The planner didn't finish in time (this happens on instances 6 and 9).
        return None, TIMEOUT_S, "timeout"
    secs = time.time() - t0

    # Metric-FF prints "found legal plan as follows" right before the plan.
    if "found legal plan" not in out:
        return None, secs, "no plan"

    # The plan lines look like "step    0: CALIBRATE ROVER0 ..." or "        1: TAKE_IMAGE ...".
    # This regex grabs the action text after the "N:" on each line.
    steps = re.findall(r"^\s*(?:step\s+)?\d+:\s+(.+)$", out.split("found legal plan", 1)[1], re.M)

    # VAL expects one action per line, in lowercase, wrapped in brackets.
    return [f"({s.strip().lower()})" for s in steps], secs, "ok"


def run_val(problem: Path, plan: Path):
    """Ask VAL whether the plan is valid for this problem.

    Returns (is_valid, full_VAL_output). The output is kept so we can read it later.
    """
    # Equivalent to typing:  Validate -v domain.pddl instance-N.pddl instance-N.plan
    # (-v = verbose, so VAL explains what it checked and, on failure, why it failed)
    out = subprocess.run([str(VAL), "-v", str(DOMAIN), str(problem), str(plan)],
                         capture_output=True, text=True).stdout
    return ("Plan valid" in out), out


def problem_stats(text: str) -> dict:
    """Count basic facts about one problem file (used for the Dataset section).

    A PDDL problem file has three parts we care about:
      (:objects ...)  - the things in the world (rovers, waypoints, cameras, ...)
      (:init ...)     - what is true at the start (the initial state)
      (:goal ...)     - what must be true at the end
    """
    # Cut the file text into those three parts.
    objs = text.split("(:objects", 1)[1].split("(:init", 1)[0]
    init = text.split("(:init", 1)[1].split("(:goal", 1)[0]
    goal = text.split("(:goal", 1)[1]

    def n_type(t):
        # Object lines look like "waypoint0 waypoint1 waypoint2 - Waypoint".
        # Find the lines ending in "- <type>" and count the names before the dash.
        return sum(len(line.split("-")[0].split()) for line in objs.splitlines()
                   if re.search(rf"-\s*{t}\s*$", line, re.I))

    # Starting energy of each rover, from lines like "(= (energy rover0) 50)".
    energies = [int(e) for e in re.findall(r"\(=\s*\(energy \w+\)\s*(\d+)\)", init)]

    return {
        # How many objects of each type.
        "rovers": n_type("rover"), "waypoints": n_type("waypoint"),
        "cameras": n_type("camera"), "objectives": n_type("objective"),
        # How many facts are in the initial state (each fact is one line starting with "(").
        "init_facts": len(re.findall(r"^\s*\(", init, re.M)),
        # Counts of the facts that our anomaly categories will perturb.
        "can_traverse": init.count("(can_traverse"),   # terrain hazard / blocked path
        "visible": init.count("(visible "),            # communication loss (note the space:
                                                       # this skips "visible_from")
        "in_sun": init.count("(in_sun"),               # power loss
        "soil_samples": init.count("(at_soil_sample"), # displaced object
        "rock_samples": init.count("(at_rock_sample"), # displaced object
        "energy_min": min(energies), "energy_max": max(energies),  # power loss
        # Every goal is a "communicated_..." fact (soil, rock or image data sent to the lander).
        "goals": len(re.findall(r"\(communicated_", goal)),
    }


def main():
    # Which instances to process, e.g. "python make_nominal.py 1 5" does instance-1..instance-5.
    first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) >= 3 else (1, 10)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []  # one row per problem, for the CSV

    for i in range(first, last + 1):
        name = f"instance-{i}"
        problem = DATA / f"{name}.pddl"

        # Start the row with the problem's statistics.
        row = {"problem": name, **problem_stats(problem.read_text())}

        # Try to solve it. Plan columns start empty and are filled in only if a plan was found.
        plan, secs, status = run_ff(problem)
        row.update(planner_status=status, planner_s=round(secs, 3), plan_len="",
                   n_navigate="", n_recharge="", val_valid="")

        if plan:
            # Save the plan, e.g. results/nominal/instance-1.plan
            plan_path = OUT / f"{name}.plan"
            plan_path.write_text("\n".join(plan) + "\n")

            # Check it with VAL and keep VAL's full report, e.g. results/nominal/instance-1.val.log
            valid, log = run_val(problem, plan_path)
            (OUT / f"{name}.val.log").write_text(log)

            # Record plan length, how many moves and recharges it uses, and whether VAL accepted it.
            row.update(plan_len=len(plan), val_valid=valid,
                       n_navigate=sum(a.startswith("(navigate") for a in plan),
                       n_recharge=sum(a.startswith("(recharge") for a in plan))

        # One progress line per problem, e.g. "instance-1: ok len=10 valid=True 0.006s"
        print(f"{name}: {status} len={row['plan_len']} valid={row['val_valid']} {row['planner_s']}s")
        rows.append(row)

    # Write all rows to results/nominal/summary.csv (column names = the row's keys).
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
