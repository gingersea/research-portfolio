"""Archive the completed local calibration without exporting private conversations."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

STUDY = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    for name in ['environment.json', 'checksums.json', 'handoff_receipt.json']:
        if (STUDY / name).exists():
            raise FileExistsError('Refuse to replace final package metadata')
    gate = json.loads((STUDY / 'judge_gate_v2.json').read_text())
    verification = json.loads((STUDY / 'verification.json').read_text())
    assert gate['judge_passed'] and gate['measurement_passed'] and verification['passed']
    environment = {'python': platform.python_version(), 'executable': sys.executable, 'libraries': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'opencv-python', 'Pillow', 'matplotlib', 'pytest']}, 'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0], 'judge': {'actual_alias': 'gpt-6-astra', 'effort': 'high', 'dated_provider_version': None}, 'scope': 'CPU procedural rendering/measurement; no downloaded generator, no human labels, no public upload.'}
    (STUDY / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    files = sorted(path for path in STUDY.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.name not in {'handoff_receipt.json', 'checksums.json'})
    checksums = {str(path.relative_to(STUDY)): sha(path) for path in files}
    (STUDY / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path('/data/AI_Research/shared/handoff') / f'world_model_principia_gravity_{stamp}'
    shutil.copytree(STUDY, target, ignore=shutil.ignore_patterns('__pycache__'))
    for name, expected in checksums.items():
        assert sha(target / name) == expected
    assert sha(target / 'checksums.json') == sha(STUDY / 'checksums.json')
    receipt = {'destination': str(target), 'verified_files': len(checksums), 'checksums_sha256': sha(STUDY / 'checksums.json'), 'data_scope': 'Procedural controls and post-judgment analytic truth. No full session logs, conversations, reasoning or credentials.'}
    (STUDY / 'handoff_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
