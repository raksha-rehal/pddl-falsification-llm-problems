# Data

`rovers-numeric/` — IPC-3 (2002) Rovers, Numeric track, "automatic planners" version, used as-is.
- Source: https://github.com/potassco/pddl-instances, path `ipc-2002/domains/rovers-numeric-automatic`, commit `cf19edf7c53d1540ddbb396c642595e0926ee552`.
- `domain.pddl` = original `NumRover.pddl`; `instance-N.pddl` = original `pfileN` (N = 1..20). See `rovers-numeric/README.md`.
- Predicate/function names checked against the CLAUDE.md failure-category table: all present
  (`can_traverse`, `at_soil_sample`, `at_rock_sample`, `in_sun`, `energy`, `recharge`, `visible`,
  `equipped_for_{soil,rock}_analysis`, `equipped_for_imaging`, `on_board`, `calibration_target`).
- Note: `navigate` also requires `(visible ?from ?to)`, so removing a `visible` fact can block movement,
  not only communication with the lander.
