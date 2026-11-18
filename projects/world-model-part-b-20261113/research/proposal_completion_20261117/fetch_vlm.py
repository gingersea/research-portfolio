"""Pinned public visual checkpoint; isolated local cache, no service changes."""
import hashlib
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REV = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'


def main():
    target = Path('/data/AI_Research/shared/proposal_completion_20261117/qwen25vl7b')
    target.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f'https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct/revision/{REV}?blobs=true', timeout=60)
    meta.raise_for_status()
    data = meta.json()
    assert data['sha'] == REV and not data.get('gated')
    (ROOT / 'vlm_source_metadata.json').write_text(json.dumps(data, indent=2) + '\n')
    files = [r for r in data['siblings'] if r['rfilename'].endswith(('.json', '.txt', '.jinja', '.safetensors'))]
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', revision=REV,
                      allow_patterns=[r['rfilename'] for r in files], local_dir=target, max_workers=4)
    records = []
    for entry in files:
        path = target / entry['rfilename']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        expected = entry.get('lfs', {}).get('sha256')
        if expected:
            assert actual == expected, path
        records.append({'file': entry['rfilename'], 'sha256': actual,
                        'expected_lfs_sha256': expected, 'bytes': path.stat().st_size})
    receipt = {'repo': 'Qwen/Qwen2.5-VL-7B-Instruct', 'revision': REV,
               'directory': str(target), 'files': records, 'external_inference': False,
               'protocol_sha256': hashlib.sha256((ROOT / 'PROTOCOL.md').read_bytes()).hexdigest()}
    (ROOT / 'vlm_assets.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'download_complete': True, 'files': len(records), 'bytes': sum(r['bytes'] for r in records)}))


if __name__ == '__main__':
    main()
