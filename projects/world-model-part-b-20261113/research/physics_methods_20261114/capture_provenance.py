"""Capture source manifests and freshly verify the reused base-model weights."""
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
CACHE = Path('/data/AI_Research/shared/physics_methods_20261114')
MODEL = Path('/home/room115/jhy/world-model-physical-consistency/.worktrees/video-bench/benchmark/physcheck/models_cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    receipt = json.loads((STUDY.parent / 'part_b_20261113T114040Z/weights_verification.json').read_text())
    rows = []
    for item in receipt['weights']:
        if item['repo'] != 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers':
            continue
        actual = sha(MODEL / item['file'])
        assert actual == item['sha256'], item['file']
        rows.append({**item, 'bytes': (MODEL / item['file']).stat().st_size, 'freshly_verified': True})
    assert len(rows) == 8
    configs = [{'file': str(path.relative_to(MODEL)), 'sha256': sha(path)} for path in sorted(MODEL.rglob('*')) if path.is_file() and path.suffix in {'.json', '.model'} and not any(part in {'assets', 'examples', '.cache'} for part in path.relative_to(MODEL).parts)]
    (STUDY / 'base_model_manifest.json').write_text(json.dumps({'repo': rows[0]['repo'], 'revision': rows[0]['revision'], 'local_dir': str(MODEL), 'weights': rows, 'local_config_tokenizer_hashes': configs, 'config_provenance': 'Observed local bytes; upstream revision/hash verification above applies to eight weight shards. Local configuration hashes retained separately.'}, indent=2) + '\n')
    for name in ['vpt', 'videophy']:
        source = json.loads((CACHE / 'source' / name / 'SOURCE_MANIFEST.json').read_text())
        for item in source['files']:
            data = (Path(source['local_dir']) / item['path']).read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert actual == item['git_blob'], item['path']
        (STUDY / f'source_{name}_manifest.json').write_text(json.dumps(source, indent=2) + '\n')
    print(json.dumps({'base_weight_files_verified': len(rows), 'local_config_files': len(configs), 'official_source_exports_verified': 2}))


if __name__ == '__main__':
    main()
