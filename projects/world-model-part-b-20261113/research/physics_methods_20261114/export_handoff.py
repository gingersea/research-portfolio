"""Package verified study artifacts in a new local shared handoff directory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

STUDY = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    audit = json.loads((STUDY / 'generation_audit.json').read_text())
    assert audit['production_videos'] == 6 and not audit['physical_improvement_claim']
    assert '待6段' not in (STUDY / 'REPORT.zh.md').read_text()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = Path('/data/AI_Research/shared/handoff') / f'world_model_physics_methods_{stamp}'
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(STUDY.rglob('*')):
        relative = path.relative_to(STUDY)
        if not path.is_file() or any(part in {'__pycache__', 'smoke'} for part in relative.parts) or path.name == 'handoff_receipt.json':
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        before = sha(path)
        assert sha(target) == before
        records.append({'path': str(relative), 'bytes': path.stat().st_size, 'sha256': before})
    # Include external public annotations without changing the source dataset.
    data = json.loads((STUDY / 'dataset_manifest.json').read_text())
    for name in ['videophy2_test.csv', 'README.md']:
        original = Path(data['local_dir']) / name
        target = destination / 'published_data' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(original, target)
        assert sha(target) == sha(original)
        records.append({'path': str(target.relative_to(destination)), 'bytes': target.stat().st_size, 'sha256': sha(target)})
    (destination / 'PACKAGE_MANIFEST.json').write_text(json.dumps({'source_study': str(STUDY), 'created_utc': stamp, 'files': records}, indent=2) + '\n')
    receipt = {'destination': str(destination), 'files': len(records), 'bytes': sum(r['bytes'] for r in records), 'package_manifest_sha256': sha(destination / 'PACKAGE_MANIFEST.json')}
    (STUDY / 'handoff_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
