"""Versioned local handoff; verify source preservation and every copied artifact."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from diagnose import sha

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
OLD = ROOT.parent / 'principia_i2v_20261115'


def main():
    receipt_path = ROOT / 'handoff_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Delivery already exists; use a new version')
    before = json.loads((ROOT / 'original_before.sha256.json').read_text())
    assert all(sha(OLD / name) == digest for name, digest in before.items())
    # Check local report links before making the delivery immutable.
    checked = []
    for path in [ROOT / 'REPORT.zh.md', ROOT / 'READING_2D_TO_3D.zh.md']:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if not link.startswith(('http:', 'https:', '#')):
                destination = (path.parent / link.split('#')[0]).resolve()
                assert destination.exists(), (path, link)
                checked.append(str(destination))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path('/data/AI_Research/shared/handoff') / f'world_model_tracking_audit_20261115_v1_{stamp}'
    target.mkdir(parents=True, exist_ok=False)
    for study in [ROOT, OLD]:
        destination = target / 'research' / study.name
        shutil.copytree(study, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for path in study.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                assert sha(path) == sha(destination / path.relative_to(study))
    for relative in [
        'research/principia_gravity_20261114/measure.py',
        'research/principia_gravity_20261114/source_snapshots/heldout_v2/manifest.json',
        'research/physics_methods_20261114/fetch_ranges.py',
        'docs/TRACKING_AUDIT_2026-11-15.zh.md',
    ]:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, destination)
    shutil.copy2('/data/AI_Research/shared/research_plans/world_model_tracking_audit_20261115.md', target / 'RESEARCH_PLAN.md')
    (target / 'HANDOFF.md').write_text(
        '# Tracking audit and 2D-to-3D reading\n\n'
        'Read research/tracking_audit_20261115/REPORT.zh.md and READING_2D_TO_3D.zh.md. '
        'The marker cardinality limitation is corrected in a separate prototype. '
        'The object prototype improves one clip but FAILS the additional identity replacement challenge; '
        'do not promote it to physics scoring. Original results remain unchanged.\n\n'
        'Contains all four original generated clips/quantized arrays, 64 known-path controls, '
        'the identity counterexample, code/diff, logs, versioned primary paper sources and notes. '
        'No generator weights, conversation transcripts or credentials. Local handoff only. '
        'Original predecessor receipts retain historical paths. Do not rerun scripts into existing output directories.\n\n'
        'MANIFEST.sha256.json covers every delivered file except itself. '
        'The source study handoff_receipt.json records its hash.\n')
    manifest = {str(path.relative_to(target)): sha(path) for path in sorted(target.rglob('*')) if path.is_file()}
    manifest_path = target / 'MANIFEST.sha256.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(sha(target / name) == digest for name, digest in manifest.items())
    result = {'handoff_directory': str(target), 'manifest_sha256': sha(manifest_path),
              'verified_files': len(manifest), 'verified_local_report_links': len(checked),
              'bytes': sum((target / name).stat().st_size for name in manifest),
              'original_files_unchanged': len(before), 'all_copy_hashes_passed': True,
              'candidate_identity_challenge': 'FAILED; retained as an explicit scientific limitation',
              'production_expansion': False, 'public_upload': False, 'created_utc': stamp}
    receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
