"""Versioned local handoff of CPU-reproducible inputs and analysis."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from common import PAPER, ROOT, save, sha


def main():
    handoff = Path('/data/AI_Research/shared/handoff') / (
        'world_model_measurement_validation_20261115_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    handoff.mkdir(exist_ok=False)
    project = ROOT.parents[1]
    shutil.copytree(ROOT, handoff / 'research' / ROOT.name,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    copied_paper = handoff / 'research' / PAPER.name
    copied_paper.mkdir()
    for path in PAPER.iterdir():
        if path.is_file() and path.suffix in ['.py', '.json', '.md']:
            shutil.copy2(path, copied_paper / path.name)
    for name in ['data', 'predictions', 'pilot3d', 'pilot3d_predictions', 'results', 'pilot3d_results']:
        shutil.copytree(PAPER / name, copied_paper / name)
    shutil.copy2(project / 'README.md', handoff / 'README.original.md')
    shutil.copy2(project / 'docs/MEASUREMENT_VALIDATION_2026-11-15.zh.md', handoff / 'MEASUREMENT_VALIDATION.zh.md')
    shutil.copy2('/data/AI_Research/shared/research_plans/world_model_measurement_validation_20261115.md', handoff / 'SHARED_PLAN.md')
    (handoff / 'README.md').write_text(
        '# Existing-label measurement validation\n\n'
        'Main report: research/measurement_validation_20261115/REPORT.zh.md\n'
        'This bundle contains frozen model predictions, paper inputs/truth, scoring sources, '
        'calibration split/receipts, tables, figures, tests and source snapshots. '
        'Previous neural inference is not rerun by this study. Raw release archives/model weights '
        'remain in the previous shared study with pinned provenance. Python/ML environment is external.\n\n'
        'Local CPU reproduction in an unused directory:\n\n'
        '    /data/AI_Research/shared/paper_benchmark_20261115/env/bin/python '
        'research/measurement_validation_20261115/verify.py reproduce --bundle "' + str(handoff) + '"\n\n'
        'The command refuses to overwrite reproduction_run. See REPRODUCTION.json for the actual executed copy.\n')
    hashes = {}
    for path in sorted(handoff.rglob('*')):
        if path.is_file():
            relative = str(path.relative_to(handoff))
            hashes[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
            if relative.startswith('research/' + ROOT.name + '/'):
                original = ROOT / path.relative_to(handoff / 'research' / ROOT.name)
                assert sha(original) == hashes[relative]['sha256']
            elif relative.startswith('research/' + PAPER.name + '/'):
                original = PAPER / path.relative_to(copied_paper)
                assert sha(original) == hashes[relative]['sha256']
    save(handoff / 'MANIFEST.json', {'files': hashes, 'copy_verified': True,
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'matplotlib', 'pytest']},
        'python': sys.version, 'utc': datetime.now(timezone.utc).isoformat()})
    save(ROOT / 'handoff_receipt.json', {'path': str(handoff), 'files_at_copy': len(hashes),
         'bytes_at_copy': sum(r['bytes'] for r in hashes.values()), 'all_copied_files_verified': True})
    print(json.dumps({'path': str(handoff), 'files': len(hashes), 'bytes': sum(r['bytes'] for r in hashes.values())}))


if __name__ == '__main__':
    main()
