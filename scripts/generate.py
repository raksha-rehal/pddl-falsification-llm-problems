"""RQ1 pipeline step 1: ask the LLM for perturbed problem files.

Talks to a vLLM OpenAI-compatible server (see scripts/serve_qwen.sh). For every
(instance, category, variant, sample) it sends one chat request and appends the raw answer
to results/generations/<run>/raw.jsonl. Already-finished jobs are skipped, so the script can
be re-run to resume after an interruption.

Usage:
  python scripts/generate.py --run qwen3.8-27b_thinking   --thinking
  python scripts/generate.py --run qwen3.8-27b_nothinking
Options: --instances 1 2 ... (default: all instances with a VAL-valid nominal plan)
         --categories ... --variants zero_shot few_shot --samples 3 --limit N (pilot runs)
"""
import argparse
import csv
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prompts import CATEGORIES, build_messages

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "rovers-numeric"
NOMINAL = ROOT / "results" / "nominal"

# Sampling settings recommended on the Qwen3.8-27B model card (Best Practices).
SAMPLING = {
    True:  dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),   # thinking
    False: dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5),   # non-thinking
}


def nominal_instances():
    """Instances whose nominal plan VAL accepted (from results/nominal/summary.csv)."""
    with open(NOMINAL / "summary.csv") as f:
        return [int(r["problem"].split("-")[1]) for r in csv.DictReader(f) if r["val_valid"] == "True"]


def call(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output folder name under results/generations/")
    ap.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    ap.add_argument("--reasoning-effort", default="xhigh", choices=["xhigh", "medium", "low"])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--instances", type=int, nargs="*")
    ap.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    ap.add_argument("--variants", nargs="*", default=["zero_shot", "few_shot"])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, help="only run the first N jobs (pilot)")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "generations" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "raw.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add(r["job_id"])

    # Build the job list: one request per (instance, category, variant, sample).
    jobs = []
    for inst in args.instances or nominal_instances():
        problem = (DATA / f"instance-{inst}.pddl").read_text()
        plan = (NOMINAL / f"instance-{inst}.plan").read_text()
        for cat in args.categories:
            for var in args.variants:
                for s in range(args.samples):
                    job_id = f"instance-{inst}__{cat}__{var}__s{s}"
                    if job_id not in done:
                        jobs.append((job_id, inst, cat, var, s, build_messages(problem, plan, cat, var)))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(done)} already done, {len(jobs)} to run -> {out_file}")

    settings = {"model": args.model, "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort if args.thinking else None,
                "max_tokens": args.max_tokens, **SAMPLING[args.thinking]}
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    lock = threading.Lock()

    def run(job):
        job_id, inst, cat, var, s, msgs = job
        sp = SAMPLING[args.thinking]
        body = {"model": args.model, "messages": msgs, "max_tokens": args.max_tokens,
                "seed": 1000 * inst + 10 * list(CATEGORIES).index(cat) + s,
                "temperature": sp["temperature"], "top_p": sp["top_p"],
                "presence_penalty": sp["presence_penalty"],
                "top_k": sp["top_k"], "min_p": sp["min_p"],
                "chat_template_kwargs": {"enable_thinking": args.thinking}}
        if args.thinking:
            # The chat template reads reasoning_effort as a template variable; pass it both ways.
            body["reasoning_effort"] = args.reasoning_effort
            body["chat_template_kwargs"]["reasoning_effort"] = args.reasoning_effort
        rec = {"job_id": job_id, "instance": f"instance-{inst}", "category": cat,
               "variant": var, "sample": s, "thinking": args.thinking}
        t0 = time.time()
        try:
            resp = call(args.url, body, timeout=3 * 3600)
            msg = resp["choices"][0]["message"]
            rec.update(content=msg.get("content") or "",
                       reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                       finish_reason=resp["choices"][0].get("finish_reason"),
                       usage=resp.get("usage", {}))
        except Exception as e:          # keep going; failed jobs are retried on the next run
            rec["error"] = repr(e)
        rec["seconds"] = round(time.time() - t0, 1)
        with lock:
            with open(out_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(args.concurrency) as pool:
        for i, fut in enumerate(as_completed([pool.submit(run, j) for j in jobs]), 1):
            r = fut.result()
            toks = r.get("usage", {}).get("completion_tokens")
            print(f"[{i}/{len(jobs)}] {r['job_id']} {r.get('finish_reason', r.get('error'))} "
                  f"{toks} tok {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
