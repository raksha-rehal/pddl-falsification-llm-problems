# Nominal plans (pipeline step 0)

Produced by `scripts/make_nominal.py` (Metric-FF v2.1, default search, 300 s limit; checked with VAL).

- Attempted: instance-1 … instance-10.
- Kept (VAL-valid nominal plan): 1, 2, 3, 4, 5, 7, 8, 10.
- Excluded: **6 and 9** — no plan within 300 s with Metric-FF (all 6 search configs) or ENHSP-20
  (sat-hmrphj and sat-hadd, with and without the recharges metric). Not proven unsolvable.

## TODO (after Milestone 2)
- Retry instances 6 and 9 with a longer time limit (e.g. 30–60 min) in the background, keeping the
  same criteria: unchanged IPC problem file, planner-found plan, VAL-valid. If found, add them and
  report the per-instance time limit.

## Notes
- ENHSP (tools/enhsp) needs two workarounds, applied only to copies: lowercase the problem files
  (it is case-sensitive about type names), and drop the paired `(not (available ?r))`/`(available ?r)`
  and `(not (channel_free ?l))`/`(channel_free ?l)` effects in the communicate actions (it otherwise
  applies the delete last and reports instance-1 unsolvable).
