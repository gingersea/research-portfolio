# Reproduction and exact scope

Run from the repository root. Use new output directories. No script overwrites
an old experiment. `tests_final.json` records the actual test interpreter and
versions. Local VLM inference used the separate existing
`/data/AI_Research/shared/physics_methods_20261114/env_vpt/bin/python`
(Transformers4.57.6), while the public tracking environment inherits
Transformers5.20.0. Do not interchange these environments without validation.

## Public inference and cached replay

```bash
PYTHONPATH=benchmark /data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  -m physcheck.relations --manifest research/proposal_completion_20261117/public_manifest.json \
  --domain davis --fresh-public --gpu 0 --output /tmp/physcheck-davis-new

PYTHONPATH=benchmark /data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  -m physcheck.relations --manifest research/proposal_completion_20261117/public_manifest.json \
  --domain pstudio --fresh-public --gpu 1 --output /tmp/physcheck-pstudio-new

PYTHONPATH=benchmark /data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  -m physcheck.relations --manifest research/proposal_completion_20261117/public_full_rescore_manifest.json \
  --output /tmp/physcheck-rescore-new

/data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  research/proposal_completion_20261117/build_reports.py --output /tmp/physcheck-report-new

/data/AI_Research/shared/paper_benchmark_20261115/env/bin/python \
  research/proposal_completion_20261117/plot_results.py --output /tmp/physcheck-figures-new
```

`public_manifest.json` binds all33 inputs, labels and frozen modules. `--fresh-public`
runs native networks; omitting it needs prediction/receipt entries, as supplied
by the rescore manifest. Public datasets do not require a physical judge gate.
The inherited old procedural judge-calibration block in original manifests is
an unrelated historical control check, not evidence validating the new Qwen
judge. New judge validity comes only from this study's failed new gates.

The final handoff contains `copied_public_manifest.json` and a recorded actual
33-case replay from copied source and copied inputs/predictions. That manifest
omits the unrelated historical gravity-judge block and retains only the public
metric source. Its `COPY_REPLAY.json` compares every per-case evaluation and
normalized coordinate/visibility array to the frozen new runs, preserving NaN
positions. It does not rerun neural inference. See `handoff_receipt.json` for
the versioned directory and all verified copied-file hashes.

## Judge inputs, model and first responses

`human_probe_plan.json` fixes all24 video IDs, original URLs, human labels and
splits before judging. `human_videos/` and `human_videos_direct/` keep original
retrieval attempts; input sheets and first responses are in `judge_inputs_pc_v2/`
and `pc_judgments/`. Gravity/I2V inputs and responses use `judge_inputs/` and
`gravity_judgments/`. `run_visual_judge.py` validates weight/image hashes and
starts a new prompt for every case. It reads labels only through the later
analysis script, never in inference context. Local inference is free of API fees.

The Qwen model snapshot is external to the code package at
`/data/AI_Research/shared/proposal_completion_20261117/qwen25vl7b_ms`.
`vlm_assets.json` records each file's hash and the frozen HF revision; each
large ModelScope-mirror file was checked against that HF revision's LFS SHA256.
Partial first downloads and range chunks remain outside the package; they are
not reclassified as complete assets. The package is not a clean-machine install
or a full offline neural-model bundle.

Preparers, downloads, judge runners and `analyze_new.py` use fixed dated output
names and reject overwrite. To repeat those stages, create a new study copy and
explicitly relocate dependencies/manifests first; do not run them over this study.
The existing receipts are the evidence of this round, not commands advertised
as safe in-place reruns. Do not rejudge failed controls to turn this test into a
confirmation set. No external API or Google service is required.

## Local delivery

`build_handoff.py` creates a new timestamped directory, copies and hashes the
stated sources/results, and actually replays the public cases and regenerates
the three reports. External neural weights/environments and historical studies
are documented dependencies. Raw author videos are kept for local research;
their redistribution rights are not inferred from the annotation-table license.
No new public upload, git push or remote verification was done. The old plan's
gingersea destination conflicts with current `/data/AI_Research/AGENTS.md`.
