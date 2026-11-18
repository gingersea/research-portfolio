# Part B Judging Workflow (interactive Claude-as-judge)

The Part B judge runs as an interactive session: a local agent (Claude, via
its Read tool) views frame grids and answers the fixed protocol template.
This file documents the exact loop so the process is reproducible.

## Prerequisites

1. Part B and calibration videos generated:
   ```
   PYTHONPATH=benchmark python3 -m physcheck generate --prompt-set part_b
   PYTHONPATH=benchmark python3 -m physcheck generate --prompt-set calibration
   ```
2. Frame grids sampled for every video:
   ```
   PYTHONPATH=benchmark python3 -m physcheck sample <video.mp4> --video-id partB_<prompt_id>_<seed>
   ```
   Video IDs are opaque: `partB_{prompt_id}_{seed}` and `calib_{prompt_id}_{seed}`.
   The model identity is NOT part of the video ID (blindness).

## Judging loop

For each video (calibration videos FIRST):
1. Read the grid image `artifacts/frames/<video_id>/grid.png` with the Read tool.
2. Answer in the JSON schema from `physcheck/judge/protocol.py`.
3. Append one line to `results/raw/judge_outputs.jsonl`:
   ```json
   {"video_id": "partB_col_b1_0", "raw_output": "<the judge's JSON answer as a string>", "judge_model": "claude", "judge_version": "<model id>"}
   ```

## Validation checkpoint

```
PYTHONPATH=benchmark python3 -m physcheck judge \
  --input-file benchmark/results/raw/judge_outputs.jsonl \
  --output-file benchmark/results/raw/judge_validated.jsonl
```
This runs the calibration gate: every calibration video must be flagged with
its expected violation kind, otherwise the protocol is invalid and the missed
videos must be re-judged.

## Swapping the judge backend

The protocol is backend-agnostic. To re-run with GPT-4o/Gemini, set
`OPENAI_API_KEY`/`GEMINI_API_KEY` and answer the same prompt template via API;
then run the same `judge` validation command. This month no paid API calls are
made (zero-budget constraint).
