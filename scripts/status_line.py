#!/usr/bin/env python3
"""Claude Code status line: progress of the RQ1 generation runs.

Shows, for each run in results/generations/ (pilot runs starting with "_" are skipped):
answers done / total, a small bar, and an ETA from the pace so far; plus request errors
and whether the vLLM server answers. Example:
  RQ1  nothinking 120/240 ██████░░░░ ETA 0h31m | thinking 45/240 ██░░░░░░░░ ETA 2h10m | err 0 | vLLM up
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
TOTAL = 240          # 8 instances x 5 categories x 2 variants x 3 samples
BAR = 10


def server_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=0.5)
        return True
    except Exception:
        return False


def run_status(run_dir: Path):
    raw = run_dir / "raw.jsonl"
    done, errors = set(), 0
    if raw.exists():
        for line in raw.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                errors += 1
            else:
                done.add(r["job_id"])
    n = len(done)
    bar = "█" * round(BAR * n / TOTAL) + "░" * (BAR - round(BAR * n / TOTAL))
    name = run_dir.name.split("_", 1)[-1]          # "qwen3.8-27b_thinking" -> "thinking"
    text = f"{name} {n}/{TOTAL} {bar}"
    start = (run_dir / "settings.json").stat().st_mtime if (run_dir / "settings.json").exists() else None
    if n >= TOTAL:
        text += " done"
    elif n and start:
        secs = (time.time() - start) / n * (TOTAL - n)
        text += f" ETA {int(secs // 3600)}h{int(secs % 3600 // 60):02d}m"
    return text, errors


def main():
    try:
        sys.stdin.read()                           # Claude Code sends session JSON; not needed
    except Exception:
        pass
    runs = sorted(d for d in GEN.glob("*") if d.is_dir() and not d.name.startswith("_")) if GEN.exists() else []
    if not runs:
        print("RQ1  no runs yet")
        return
    parts, errs = [], 0
    for d in runs:
        t, e = run_status(d)
        parts.append(t)
        errs += e
    print("RQ1  " + " | ".join(parts) + f" | err {errs} | vLLM {'up' if server_up() else 'DOWN'}")


if __name__ == "__main__":
    main()
