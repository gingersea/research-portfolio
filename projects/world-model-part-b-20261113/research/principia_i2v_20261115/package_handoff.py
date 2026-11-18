"""Create a new local handoff with copied dependencies and verified file hashes."""
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

STUDY = Path(__file__).resolve().parent
PROJECT = STUDY.parent.parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = STUDY / 'handoff_receipt.json'
    if receipt.exists():
        raise FileExistsError('Handoff already recorded; use a new version for a new delivery')
    versions = {name: importlib.metadata.version(name) for name in [
        'torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'huggingface-hub',
        'tokenizers', 'safetensors', 'numpy', 'scipy', 'Pillow', 'opencv-python', 'requests']}
    assert versions['diffusers'] == '0.36.0' and versions['transformers'] == '4.57.6', 'Use the inference interpreter'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path('/data/AI_Research/shared/handoff') / f'world_model_principia_i2v_20261115_v1_{stamp}'
    target.mkdir(parents=True, exist_ok=False)
    study_copy = target / 'research' / STUDY.name
    shutil.copytree(STUDY, study_copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    dependencies = [
        'research/principia_gravity_20261114/measure.py',
        'research/principia_gravity_20261114/source_snapshots/heldout_v2/manifest.json',
        'research/physics_methods_20261114/fetch_ranges.py',
        'docs/PRINCIPIA_I2V_2026-11-15.zh.md',
    ]
    for relative in dependencies:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, destination)
        assert sha(destination) == sha(PROJECT / relative)
    shutil.copy2('/data/AI_Research/shared/research_plans/world_model_principia_i2v_20261115.md', target / 'RESEARCH_PLAN.md')
    (target / 'INFERENCE_ENVIRONMENT.json').write_text(json.dumps({
        'python': platform.python_version(), 'versions': versions,
        'interpreter': '/data/AI_Research/shared/physics_methods_20261114/env_vpt/bin/python',
        'torch_runtime': '2.9.0+cu128', 'cuda_build': '12.8',
        'analysis_environment_record': f'research/{STUDY.name}/summary/artifact_verification.json',
    }, indent=2) + '\n')
    snapshots = sorted((STUDY / 'source_snapshots').glob('*/manifest.json'), key=lambda p: json.loads(p.read_text())['utc'])
    differences = []
    for before_manifest, after_manifest in zip(snapshots, snapshots[1:]):
        before, after = before_manifest.parent, after_manifest.parent
        names = sorted({p.name for p in before.iterdir() if p.suffix in ['.py', '.md']} | {p.name for p in after.iterdir() if p.suffix in ['.py', '.md']})
        for name in names:
            old = (before / name).read_text().splitlines(keepends=True) if (before / name).exists() else []
            new = (after / name).read_text().splitlines(keepends=True) if (after / name).exists() else []
            differences.extend(difflib.unified_diff(old, new, fromfile=f'{before.name}/{name}', tofile=f'{after.name}/{name}'))
    (target / 'SOURCE_HISTORY.patch').write_text(''.join(differences))
    (target / 'HANDOFF.md').write_text(
        '# Native I2V pilot local delivery\n\n'
        'Read research/principia_i2v_20261115/REPORT.zh.md first. Four fixed attempts completed; '
        'initial preservation 4/4, coverage 0/4, physics relations 0/4 (N/A), expansion stopped.\n\n'
        'All study inputs, quantized native frame arrays, videos, metrics, plots, source history, '
        'review and failure logs are included. Legacy dependencies are copied read-only. '
        'Generator weights are excluded: retrieve the immutable revision and verify native_assets_complete.json. '
        'No conversation transcripts, credentials or raw reasoning are included. This is a local delivery, not publication.\n\n'
        'MANIFEST.sha256.json hashes every delivered file except itself. '
        'Original study handoff_receipt.json records the manifest hash. '
        'Generation/evaluation commands in the report describe executed runs; use new output paths for replay.\n')
    manifest = {str(path.relative_to(target)): sha(path) for path in sorted(target.rglob('*')) if path.is_file()}
    manifest_path = target / 'MANIFEST.sha256.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(sha(target / name) == digest for name, digest in manifest.items())
    # Verify copying against original study as well, independently of the new manifest.
    original_files = [p for p in STUDY.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    assert all(sha(path) == sha(study_copy / path.relative_to(STUDY)) for path in original_files)
    result = {'handoff_directory': str(target), 'manifest_sha256': sha(manifest_path),
              'verified_files': len(manifest), 'original_study_files_equal': len(original_files),
              'bytes': sum((target / name).stat().st_size for name in manifest),
              'all_file_hashes_passed': True, 'public_upload': False,
              'generator_weights_included': False, 'created_utc': stamp}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
