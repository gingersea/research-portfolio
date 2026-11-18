# Closeout and provisional action-world pivot handoff

Canonical synthesis: `docs/CLOSEOUT_AND_PIVOT_2026-11-14.zh.md`.

The former text-to-video physics project is closed as a feasibility/methodology
study, retaining its negative results and incomplete scientific goals. The active
provisional direction is action-conditioned prediction and closed-loop planning.
The user authorized changing direction but has not specified the exact topic;
this first CPU-only stage is reversible and does not lock a final research claim.

## New results

- `research/closeout_20261114/results/`: exact50-condition exposure enumeration,
  twelve lossless video pairs, actual sampled images/grids and full world-state
  intervention records. These are deliberate counterexamples, not prevalence or
  VLM accuracy measurements.
- `research/action_world_stage0_20261114/results/`: six least-squares state models,
  train/test arrays,768 per-trajectory prediction records,448 closed-loop runs on
  the same64 cases, and a known-dynamics matched-controller reference. This is an
  intentionally easy linear-world feasibility baseline, not a visual model.
- The original protocols and executed source are preserved with recorded hashes.
  Each study has a post-run `CLARIFICATIONS.md` for wording corrections, without
  changing equations, data, labels or results.
- `verification.json`: actual in-repository replay, including all771 preserved
  hashes of the previous Part B study. `REVIEW_RESOLUTION.md` records independent
  checks and the two nonblocking review notes.

## Verify and reproduce

Original environment: `/home/room115/pytorch_env/bin/python`; Python3.12.3,
numpy2.5.3, av17.0.1, OpenCV4.13.0 and imageio2.37.3. This round uses no torch,
model weights, paid API, GPU training or VLM judging. FFV1 support is needed in
the PyAV build for regeneration.

From the repository root or extracted handoff root:

```bash
python research/closeout_20261114/verify.py
```

The verifier checks result hashes, executed-source/protocol hashes, finite-window
counts, all24 video decodes/paired frames, six prediction summaries and all448
planning trajectories. It does not rewrite outputs. If a package manifest exists,
its source/summary hashes are checked too.

Regenerate into NEW directories only:

```bash
python research/closeout_20261114/experiment.py --output replay_closeout_NEW
python research/action_world_stage0_20261114/experiment.py --output replay_action_NEW
```

The archive contains the two new studies, minimal unchanged sampler source and
their protocols/summary. It does not duplicate all historical video weights or
earlier studies. Thus `old_part_b_checksums_preserved` is null when verifying a
standalone archive; the original in-repository receipt records771. The previous
full Part B handoff remains at
`/data/AI_Research/shared/handoff/part_b_20261113T114040Z/`.

## Next study, before substantial compute

Refine the literature difference and lock one pixel/partial-observation task;
use episode-level splits and held-out dynamics. Keep state oracle, persistence,
action-blind and a capable reactive/PD controller. Report actual goal success and
failure costs alongside prediction error. Winning only against an action-blind
B=0 model is not evidence of a useful new method. Do not relaunch the old broad
generation/calibration loop simply to obtain a passing ranking.
