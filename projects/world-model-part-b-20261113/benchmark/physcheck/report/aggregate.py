"""Result aggregation: JSONL artifacts → leaderboard tables.

Part A: per-model parameter estimates, relative errors, success rates.
Part B: violation rates by model/category from judge outputs + manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def aggregate_part_a(jsonl_lines: list[dict]) -> dict:
    """Aggregate Part A results into per-model statistics.

    Returns {"by_model": {model: {"g_rel_err_mean", "T_rel_err_mean",
    "fit_success_rate", "track_success_rate", "per_prompt": {...}}}}.
    """
    by_model: dict[str, dict] = {}
    for line in jsonl_lines:
        model = line["model"]
        m = by_model.setdefault(model, {"lines": [], "per_prompt": {}})
        m["lines"].append(line)

    out = {"by_model": {}}
    for model, m in by_model.items():
        lines = m["lines"]
        n = len(lines)
        n_track_ok = sum(1 for l in lines if l.get("track_status") == "ok")
        n_fit_ok = sum(1 for l in lines if l.get("fit_status") == "ok")
        g_errs = [
            l["rel_errs"]["g"]
            for l in lines
            if l.get("rel_errs") and "g" in l["rel_errs"]
        ]
        T_errs = [
            l["rel_errs"]["T"]
            for l in lines
            if l.get("rel_errs") and "T" in l["rel_errs"]
        ]
        per_prompt = {}
        for l in lines:
            pid = l["prompt_id"]
            per_prompt.setdefault(pid, {"n": 0, "fit_ok": 0, "params": []})
            per_prompt[pid]["n"] += 1
            if l.get("fit_status") == "ok":
                per_prompt[pid]["fit_ok"] += 1
                per_prompt[pid]["params"].append(l.get("params", {}))

        def _mean(xs):
            return float(np.mean(xs)) if xs else None

        out["by_model"][model] = {
            "n_videos": n,
            "track_success_rate": n_track_ok / n if n else None,
            "fit_success_rate": n_fit_ok / n if n else None,
            "g_rel_err_mean": _mean(g_errs),
            "T_rel_err_mean": _mean(T_errs),
            "per_prompt": {
                pid: {
                    "n": v["n"],
                    "fit_ok": v["fit_ok"],
                    "mean_params": (
                        {k: _mean([p.get(k) for p in v["params"] if k in p]) for k in v["params"][0]}
                        if v["params"]
                        else {}
                    ),
                }
                for pid, v in per_prompt.items()
            },
        }
    return out


def aggregate_part_b(judge_outputs: list[dict], manifest: dict[str, dict]) -> dict:
    """Aggregate Part B judge outputs into violation statistics.

    judge_outputs: list of parsed judge output dicts (video_id, violations).
    manifest: {video_id: {"model": ..., "prompt_id": ..., "seed": ...,
    "kind": ...}} — maps opaque video IDs back to models (blindness: the
    judge never sees this mapping).
    """
    per_model: dict[str, list] = {}
    per_video = []
    for out in judge_outputs:
        vid = out["video_id"]
        info = manifest.get(vid)
        if info is None:
            continue
        model = info["model"]
        violations = out.get("violations", [])
        per_model.setdefault(model, []).append(len(violations))
        cats = {}
        for v in violations:
            cats[v["category"]] = cats.get(v["category"], 0) + 1
        per_video.append(
            {
                "model": model,
                "prompt_id": info.get("prompt_id"),
                "seed": info.get("seed"),
                "kind": info.get("kind"),
                "n_violations": len(violations),
                "categories": cats,
            }
        )

    violation_rate_by_model = {
        m: sum(1 for c in counts if c > 0) / len(counts)
        for m, counts in per_model.items()
    }
    mean_violations_per_video = {
        m: float(np.mean(counts)) for m, counts in per_model.items()
    }
    violations_by_category = {}
    for v in per_video:
        m = v["model"]
        d = violations_by_category.setdefault(m, {})
        for cat, c in v["categories"].items():
            d[cat] = d.get(cat, 0) + c

    return {
        "violation_rate_by_model": violation_rate_by_model,
        "mean_violations_per_video": mean_violations_per_video,
        "violations_by_category": violations_by_category,
        "per_video": per_video,
    }


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def main(cfg, part="both") -> None:
    results_dir = Path(cfg.results_dir)
    out_dir = results_dir / "leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    part_a_path = results_dir / "raw" / "part_a.jsonl"
    if part in ("a", "both") and part_a_path.exists():
        agg_a = aggregate_part_a(_load_jsonl(str(part_a_path)))
        (out_dir / "part_a_leaderboard.json").write_text(json.dumps(agg_a, indent=2))
        print("part A leaderboard:", json.dumps(agg_a["by_model"], indent=2)[:500])

    judge_path = results_dir / "raw" / "judge_validated.jsonl"
    manifest_path = results_dir / "raw" / "part_b_manifest.json"
    status_path = results_dir / "raw" / "judge_validated.status.json"
    if part in ("b", "both") and status_path.exists() and manifest_path.exists():
        from physcheck.judge.production import build_leaderboard
        manifest = json.loads(manifest_path.read_text())
        validation = json.loads(status_path.read_text())
        if validation.get('calibration_only'):
            raise ValueError('production report requires a complete production validation')
        agg_b = build_leaderboard(validation, manifest)
        (out_dir / "part_b_leaderboard.json").write_text(json.dumps(agg_b, indent=2) + '\n')
        display = agg_b if agg_b['valid'] else agg_b['diagnostic_only']
        table = ['# Part B visible-violation summary', '',
                 f'Calibration valid: {agg_b["valid"]}. Static-frame judgments only.', '',
                 '| Model | Judged production videos | Valid violation rate |', '|---|---:|---:|']
        for model in sorted(display['violation_rate_by_model']):
            n = sum(row['model'] == model for row in display['per_video'])
            rate = f'{display["violation_rate_by_model"][model]:.4f}' if agg_b['valid'] else 'N/A (invalid calibration)'
            table.append(f'| {model} | {n} | {rate} |')
        if not agg_b['valid']:
            table += ['', 'Diagnostic-only counts are retained in JSON. They are not a valid model ranking.']
        (out_dir / "part_b_leaderboard.md").write_text('\n'.join(table) + '\n')
        print(f'Part B production records: {len(display["per_video"])}; calibration valid: {agg_b["valid"]}')
    elif part in ("b", "both") and judge_path.exists() and manifest_path.exists():
        manifest = json.load(open(manifest_path))
        if any('is_calibration' in info for info in manifest.values()):
            raise ValueError('strict production report requires its bound status checkpoint')
        agg_b = aggregate_part_b(_load_jsonl(str(judge_path)), manifest)
        (out_dir / "part_b_leaderboard.json").write_text(json.dumps(agg_b, indent=2))
        print("part B leaderboard:", json.dumps(
            {k: v for k, v in agg_b.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from physcheck.config import Config

    main(Config.load())
