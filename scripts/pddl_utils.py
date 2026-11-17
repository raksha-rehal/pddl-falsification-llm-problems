"""Small helpers for reading Rovers PDDL problem files and plans (no external dependencies).

A PDDL file is a nested list of brackets, e.g. (define (problem p) (:objects a b - t) (:init ...)).
We turn it into nested Python lists, then pull out the parts we need:
  objects: {"waypoint0": "waypoint", ...}
  init:    a set of facts like ("can_traverse", "rover0", "waypoint3", "waypoint0")
           and numeric values like ("=", "energy", "rover0") -> 50.0
  goal:    a set of goal facts
PDDL is case-insensitive, so everything is lowercased.
"""
import re


def tokenize(text: str):
    text = re.sub(r";[^\n]*", "", text.lower())          # drop comments, lowercase
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(text: str):
    """Turn PDDL text into nested lists. Raises ValueError if the brackets don't match."""
    stack, cur = [], []
    for tok in tokenize(text):
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced ')'")
            parent = stack.pop()
            parent.append(cur)
            cur = parent
        else:
            cur.append(tok)
    if stack:
        raise ValueError("unbalanced '('")
    if len(cur) != 1 or not isinstance(cur[0], list) or cur[0][:1] != ["define"]:
        raise ValueError("file is not a single (define ...) form")
    return cur[0]


def _section(tree, key):
    for part in tree[1:]:
        if isinstance(part, list) and part and part[0] == key:
            return part[1:]
    return None


def _flatten_and(expr):
    """(and a b c) -> [a, b, c]; a single fact -> [fact]."""
    if isinstance(expr, list) and expr and expr[0] == "and":
        out = []
        for e in expr[1:]:
            out.extend(_flatten_and(e))
        return out
    return [expr]


def parse_problem(text: str) -> dict:
    """Parse a problem file into its parts. Raises ValueError if it is malformed."""
    tree = parse_sexpr(text)
    header = _section(tree, "problem")
    domain = _section(tree, ":domain")
    objs_raw = _section(tree, ":objects")
    init_raw = _section(tree, ":init")
    goal_raw = _section(tree, ":goal")
    metric = _section(tree, ":metric")
    if None in (header, domain, objs_raw, init_raw, goal_raw):
        raise ValueError("missing problem/:domain/:objects/:init/:goal section")

    # ":objects a b - type c - type2" -> {a: type, b: type, c: type2}
    objects, pending, i = {}, [], 0
    while i < len(objs_raw):
        tok = objs_raw[i]
        if tok == "-":
            for name in pending:
                objects[name] = objs_raw[i + 1]
            pending, i = [], i + 2
        else:
            pending.append(tok)
            i += 1
    for name in pending:                       # untyped leftovers
        objects[name] = "object"

    facts, fluents = set(), {}
    for item in init_raw:
        if not isinstance(item, list) or not all(isinstance(x, str) for x in item[:1]):
            raise ValueError(f"bad init item: {item}")
        if item[0] == "=":                     # (= (energy rover0) 50)
            lhs, val = item[1], item[2]
            if not isinstance(lhs, list) or not isinstance(val, str):
                raise ValueError(f"bad numeric init: {item}")
            fluents[tuple(lhs)] = float(val)
        else:
            if not all(isinstance(x, str) for x in item):
                raise ValueError(f"bad init fact: {item}")
            facts.add(tuple(item))

    goals = set()
    for g in _flatten_and(goal_raw[0] if goal_raw else []):
        if not isinstance(g, list) or not all(isinstance(x, str) for x in g):
            raise ValueError(f"unsupported goal: {g}")
        goals.add(tuple(g))

    return {"name": header[0] if header else None, "domain": domain[0] if domain else None,
            "objects": objects, "facts": facts, "fluents": fluents, "goals": goals,
            "metric": metric}


def fmt_fact(f) -> str:
    return "(" + " ".join(f) + ")"


def diff_problems(nominal: dict, perturbed: dict) -> dict:
    """What changed in the initial state (and whether anything outside it changed)."""
    removed = sorted(nominal["facts"] - perturbed["facts"])
    added = sorted(perturbed["facts"] - nominal["facts"])
    fl_changed = []
    for k in sorted(set(nominal["fluents"]) | set(perturbed["fluents"])):
        a, b = nominal["fluents"].get(k), perturbed["fluents"].get(k)
        if a != b:
            fl_changed.append((k, a, b))
    return {
        "removed": removed, "added": added, "fluents_changed": fl_changed,
        "n_changes": len(removed) + len(added) + len(fl_changed),
        "objects_same": nominal["objects"] == perturbed["objects"],
        "goals_same": nominal["goals"] == perturbed["goals"],
        "domain_same": nominal["domain"] == perturbed["domain"],
        "metric_same": nominal["metric"] == perturbed["metric"],
    }


def plan_actions(plan_text: str):
    """'(navigate rover0 w1 w2)' lines -> [('navigate', 'rover0', 'w1', 'w2'), ...]"""
    return [tuple(m.lower().split()) for m in re.findall(r"\(([^()]+)\)", plan_text)]
