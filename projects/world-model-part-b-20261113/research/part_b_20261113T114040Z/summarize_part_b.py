"""Render completed Part B outcomes directly from retained observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=STUDY/'run')
    parser.add_argument('--results-dir',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    results=(args.results_dir or run/'results').resolve()
    generated=[]
    for model in ['wan2.1_1.3b','cogvideox_5b']:
        generated.extend(json.loads((run/'private'/f'{model}.json').read_text()))
    board=json.loads((results/'leaderboard/part_b_leaderboard.json').read_text())
    validation=json.loads((results/'raw/judge_validated.status.json').read_text())
    controls=[r for r in generated if r['is_calibration']]
    production=[r for r in generated if not r['is_calibration']]
    assert len(controls)==8 and len(production)==48 and len(validation['validated'])==56
    parsed={r['video_id']:r for r in validation['validated']}
    report=['# Part B production and calibration results','',
            'This report is artifact-derived. Generated control prompts are not independently verified labels.', '',
            f'Checkpoint root: `{results.relative_to(STUDY)}`.', '',
            f'Calibration valid: **{board["valid"]}**. Judge: gpt-6-astra through isolated platform QA sessions.',
            'The actual dated model snapshot is not exposed by the harness. No Claude API request was made.', '',
            '## Generated matrix','', '| Model | Calibration clips | Production clips | Decoded frames | Generation status |',
            '|---|---:|---:|---:|---|']
    for model in ['wan2.1_1.3b','cogvideox_5b']:
        selected=[r for r in generated if r['model']==model]
        report.append(f'| {model} | {sum(r["is_calibration"] for r in selected)} | {sum(not r["is_calibration"] for r in selected)} | {sum(r["decoded_frames"] for r in selected)} | {sum(r["status"]=="ok" for r in selected)}/{len(selected)} decoded and hashed |')
    report+=['','Decoded and hashed means file-level generation completion, not verified prompt compliance or physical correctness.',
             'One seed per calibration prompt; three seeds per production prompt. Eight original-frame-index samples per clip.',
             'Calibration was judged before any production clip. Every clip uses a separate opaque-ID session.', '',
             '## Calibration observations','', '| Model | Control prompt | Reported categories | Eligible control? | Notes |',
             '|---|---|---|---|---|']
    manifest=json.loads((results/'raw/part_b_manifest.json').read_text())
    for record in controls:
        vid=record['video_id'];out=parsed[vid]
        cats=', '.join(v['category'] for v in out['violations']) or 'none'
        info=manifest[vid]
        notes=out['notes'].replace('|','/').replace('\n',' ')
        report.append(f'| {record["model"]} | {record["prompt_id"]} | {cats} | {info.get("calibration_observable",False)} | {notes} |')
    report+=['','A control is not a known positive merely because its prompt describes a violation.',
             'Constant uphill speed and pass-through dynamics may be unidentifiable from this static-frame protocol.',
             'No failed controls were dropped, no labels were fabricated, and no judge was coached to produce a desired answer.', '',
             '## Production leaderboard','',(results/'leaderboard/part_b_leaderboard.md').read_text(),
             'Detailed diagnostic-only per-video/category counts, if calibration is invalid, are retained in JSON rather than interpreted as a valid model ranking.', '',
             '## Proposal status','',
             '- Task 7: real calibration and production clips sampled; eight-frame grids and individual PNGs archived.',
             '- Task 8: real isolated judgments and strict validation executed. Scientific calibration validity is the gate outcome above, not assumed from code completion.',
             '- Task 10: 48 production clips and all 56 judgments retained; CLI validation and leaderboard aggregation attempted with failure status preserved.',
             '- Remaining: independently verified calibration references (including negative controls), human generated-video labels, false-positive control, reliable smallest-target measurement, valid Part A physical estimates and the complete English paper draft.', '',
             '## Interpretation and reproducibility','',
             'Each video is an evaluation unit; three seeds share a prompt and are not independent scene categories. Counts are descriptive; no P values, causal claim, or inferential confidence interval is reported.',
             'The evaluator sees eight static frames, not full temporal dynamics. An empty violation list means no observed contradiction under this protocol, not physical correctness.',
             'The frozen sampler resizes native frames to 320x180 grid cells without preserving aspect ratio. Original sampled PNGs are also archived. Wan uses 81 frames at 16 fps; CogVideoX uses 49 frames at 8 fps. Native durations differ, so diagnostic counts are not a duration-matched causal comparison.',
             'Filtered session receipts verify unique actual child UUIDs, fork_turns=none, model alias/effort and actual image-input bytes against archived grid hashes. Per-clip read scope is instruction-enforced rather than an operating-system sandbox. No full private transcripts are exported.',
             'Backend routing, rejected/failed controls and code decisions are documented in PROGRESS.md. Generator settings, video hashes, source hashes, package versions and timings remain in per-video sidecars.', '',
             'Run `generate_part_b.py --model MODEL --gpu GPU --output RUN --models-dir CHECKPOINT_CACHE` once per model. Judge controls in fresh one-clip sessions, export their filtered receipts with `export_judge_receipts.py --partial` and run `assemble_part_b.py --phase calibration` before any production judgment. After completing the production judgments, export all 56 receipts without --partial, run `audit_part_b.py`, `assemble_part_b.py --phase production`, and this renderer. Keep the original early calibration gate unchanged.',
             'The full real generated matrix must exist; scripts never fill missing judgments or generate human labels automatically.', '',
             'AI assistance: code, experiment orchestration, VLM judgments and report drafting were AI-assisted. Human truth labeling remains incomplete.', '',
             'Review findings and one TDD fix pass are documented in REVIEW_RESOLUTION.md. The initial checkpoint remains in run/results; the final bound replay is in run/results_reviewed. Both retain the same first judgments and failed scientific gate.',
             'Render the final replay with `summarize_part_b.py --results-dir research/part_b_20261113T114040Z/run/results_reviewed` from the repository root. The amended production assembler accepts the same --results-dir argument and refuses to replace a CLI checkpoint.', '',
             '## Gate details','', '```json',json.dumps(validation['calibration'],indent=2),'```']
    (STUDY/'REPORT.md').write_text('\n'.join(report)+'\n')
    files={p.relative_to(STUDY).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(STUDY.rglob('*')) if p.is_file() and p.name!='checksums.json' and '__pycache__' not in p.parts}
    (STUDY/'checksums.json').write_text(json.dumps(files,indent=2)+'\n')
    print(f'Report generated from {len(generated)} actual clips and {len(parsed)} actual judgments.')


if __name__=='__main__':main()
