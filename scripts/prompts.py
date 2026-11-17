"""Prompts for RQ1: ask the LLM to perturb a nominal Rovers problem file to inject one anomaly.

Design decisions (see the Milestone 2 methodology):
  - The LLM sees the domain, the nominal problem and the nominal plan ("white-box" falsifier).
  - Each anomaly category is described in natural language only. We do NOT tell the model which
    PDDL predicates model each anomaly, so that category agreement measures whether the model can
    map the anomaly onto the domain itself.
  - Zero-shot: instructions only. Few-shot: the same instructions plus one worked example of the
    SAME category, built from held-out instance-12 (not one of the evaluated instances), in the
    exact output format we ask for (LLM+P-style in-context example, Liu et al. 2023).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOMAIN_TEXT = (DATA / "rovers-numeric" / "domain.pddl").read_text()

# Natural-language descriptions of the five anomaly categories (CLAUDE.md, Planned Approach).
CATEGORIES = {
    "displaced_object": (
        "Displaced object: a soil or rock sample that the plan intends to collect is not where the "
        "mission expects it, because it has been displaced to a different location."),
    "blocked_path": (
        "Terrain hazard / blocked path: a terrain hazard (for example a crater rim, loose regolith or "
        "a boulder field) blocks a route between two locations that a rover relies on in the plan."),
    "power_loss": (
        "Power loss: a rover has less power available than the plan assumes, for example because of "
        "battery degradation, or because a location it relies on for solar recharging is in shadow."),
    "communication_loss": (
        "Communication loss: line of sight between a location and the lander is lost, so a rover "
        "can no longer transmit data to the lander from where the plan expects to."),
    "equipment_fault": (
        "Equipment fault: a piece of a rover's equipment that the plan uses (an analysis instrument, "
        "a camera, or a camera's calibration) has failed or is unavailable."),
}

# One-sentence descriptions of the few-shot example perturbations of instance-12
# (files in data/fewshot/, each confirmed with VAL to make the instance-12 plan fail).
FEWSHOT_CHANGES = {
    "displaced_object": "The rock sample at waypoint6 has been displaced to waypoint4.",
    "blocked_path": "A boulder field blocks rover1's route from waypoint4 to waypoint6.",
    "power_loss": "rover3's battery has degraded, so it starts with 10 energy instead of 50.",
    "communication_loss": "Line of sight from waypoint3 to the lander's waypoint (waypoint2) is lost.",
    "equipment_fault": "rover2's soil analysis instrument has failed.",
}

SYSTEM = (
    "You are a test engineer for autonomous planetary rovers. You write PDDL problem files that "
    "inject anomalies into a rover mission, so that engineers can see how a pre-computed mission plan "
    "fails when the world differs from what the plan assumed.")

INSTRUCTIONS = """\
Below are a PDDL domain for planetary rovers, a nominal problem file, and a nominal plan that is \
valid for the nominal problem.

Your task: write a perturbed copy of the nominal problem file that injects exactly one anomaly of \
this kind:

{category}

Rules:
1. Change only the (:init ...) section. Keep the problem name, (:domain ...), (:objects ...), \
(:goal ...) and (:metric ...) exactly as they are.
2. Make a small change: add, remove or modify at most 3 facts or numeric values in total.
3. Inject only the anomaly described above, not any other kind of anomaly.
4. The nominal plan must no longer work on the perturbed problem, and it should first break at \
an action that is affected by the anomaly.
5. The file must remain valid PDDL for this domain.

Answer in exactly this format and nothing else:
CHANGE: <one sentence describing the anomaly you injected>
```pddl
<the complete perturbed problem file>
```"""

TASK = """\
Domain:
```pddl
{domain}
```

Nominal problem:
```pddl
{problem}
```

Nominal plan:
```
{plan}
```"""


def _task_block(problem_text: str, plan_text: str) -> str:
    return TASK.format(domain=DOMAIN_TEXT.strip(), problem=problem_text.strip(), plan=plan_text.strip())


def build_messages(problem_text: str, plan_text: str, category: str, variant: str) -> list:
    """Chat messages for one generation request. variant is 'zero_shot' or 'few_shot'."""
    instr = INSTRUCTIONS.format(category=CATEGORIES[category])
    msgs = [{"role": "system", "content": SYSTEM}]
    if variant == "few_shot":
        ex_problem = (DATA / "rovers-numeric" / "instance-12.pddl").read_text()
        ex_plan = (DATA / "fewshot" / "instance-12.plan").read_text()
        ex_answer = (f"CHANGE: {FEWSHOT_CHANGES[category]}\n```pddl\n"
                     f"{(DATA / 'fewshot' / f'{category}.pddl').read_text().strip()}\n```")
        msgs += [{"role": "user", "content": instr + "\n\n" + _task_block(ex_problem, ex_plan)},
                 {"role": "assistant", "content": ex_answer}]
    elif variant != "zero_shot":
        raise ValueError(variant)
    msgs.append({"role": "user", "content": instr + "\n\n" + _task_block(problem_text, plan_text)})
    return msgs
