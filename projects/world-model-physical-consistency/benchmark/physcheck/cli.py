"""physcheck CLI.

Subcommands:
  generate   — generate videos for a prompt set
  detect     — Part A production driver (generate → track → fit)
  sample     — sample frames for one video (judging prep)
  judge      — validate judge outputs JSONL (schema + calibration gate)
  report     — aggregate results into leaderboard/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))


def _cmd_generate(args):
    from physcheck.config import Config
    from physcheck.generate.pipeline import generate_all

    cfg = Config.load()
    models = args.models or ["wan2.1_1.3b"]
    seeds = args.seeds or cfg.seed_grid
    generate_all(args.prompt_set, cfg, models=models, seeds=seeds)


def _cmd_detect(args):
    from physcheck.config import Config
    from physcheck.detect.run_part_a import main

    cfg = Config.load()
    main(args.models or ["wan2.1_1.3b"], cfg)


def _cmd_sample(args):
    from physcheck.config import Config
    from physcheck.judge.frames import sample_frames

    cfg = Config.load()
    out = sample_frames(args.video_path, n_frames=args.n_frames, cfg=cfg, video_id=args.video_id)
    print(json.dumps(out, indent=2))


def _cmd_judge(args):
    from physcheck.judge.calibration import check_calibration
    from physcheck.judge.protocol import parse_judge_output

    lines = [json.loads(l) for l in open(args.input_file) if l.strip()]
    outputs = {}
    for rec in lines:
        out = parse_judge_output(
            rec["raw_output"],
            rec["video_id"],
            rec.get("judge_model", "unknown"),
            rec.get("judge_version", "unknown"),
        )
        outputs[rec["video_id"]] = out
    result = check_calibration(outputs)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for vid, out in sorted(outputs.items()):
            f.write(json.dumps(out) + "\n")
    print(f"calibration: {result}")
    print(f"wrote {len(outputs)} validated outputs to {out_path}")


def _cmd_report(args):
    from physcheck.config import Config
    from physcheck.report.aggregate import main as agg_main

    cfg = Config.load()
    agg_main(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physcheck")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--prompt-set", required=True, choices=["part_a", "part_b", "calibration"])
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("detect")
    p.add_argument("--models", nargs="+")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("sample")
    p.add_argument("video_path")
    p.add_argument("--video-id", default="anon")
    p.add_argument("--n-frames", type=int, default=8)
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser("judge")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.set_defaults(func=_cmd_judge)

    p = sub.add_parser("report")
    p.add_argument("--part", choices=["a", "b", "both"], default="both")
    p.set_defaults(func=_cmd_report)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
