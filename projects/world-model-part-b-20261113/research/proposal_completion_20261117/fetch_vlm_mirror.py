"""Fast bounded direct mirror ranges, verified against the frozen HF hashes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import urljoin, urlparse

import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct/resolve/master/'
TARGET = Path('/data/AI_Research/shared/proposal_completion_20261117/qwen25vl7b_ms')
LOCAL = threading.local()
CHUNK = 8 << 20


def get(url, **kwargs):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    for _ in range(6):
        host = urlparse(url).hostname or ''
        if not (host == 'modelscope.cn' or host.endswith('.modelscope.cn')):
            raise ValueError('Unapproved mirror redirect host')
        response = LOCAL.session.get(url, allow_redirects=False, timeout=(15, 90), **kwargs)
        if response.status_code in [301, 302, 303, 307, 308]:
            url = urljoin(url, response.headers['Location'])
            response.close()
            continue
        response.raise_for_status()
        return response
    raise ValueError('Redirect loop')


def part(job):
    row, start, end = job
    name = row['rfilename']
    target = TARGET / 'parts' / name / str(start)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == end - start + 1:
        return target
    for attempt in range(3):
        try:
            response = get(BASE + name + f'?range_part={start}', headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
            assert response.status_code == 206
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{row["lfs"]["size"]}'
            content = response.content
            assert len(content) == end - start + 1
            with target.open('xb') as handle:
                handle.write(content)
            return target
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / 'vlm_source_metadata.json').read_text())
    rows = [r for r in meta['siblings'] if r['rfilename'].endswith(('.json', '.txt', '.jinja', '.safetensors'))]
    jobs = [(row, start, min(start + CHUNK, row['lfs']['size']) - 1)
            for row in rows if 'lfs' in row for start in range(0, row['lfs']['size'], CHUNK)]
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(part, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 100 == 0:
                print(json.dumps({'completed_parts': count, 'total_parts': len(jobs)}), flush=True)
    records = []
    for row in rows:
        name = row['rfilename']
        path = TARGET / name
        if 'lfs' in row:
            with path.open('xb') as handle:
                for start in range(0, row['lfs']['size'], CHUNK):
                    with (TARGET / 'parts' / name / str(start)).open('rb') as block:
                        shutil.copyfileobj(block, handle)
        else:
            original = TARGET.parent / 'qwen25vl7b' / name
            if original.exists():
                shutil.copy2(original, path)
            else:
                response = get(BASE + name)
                path.write_bytes(response.content)
        value = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                value.update(block)
        if 'lfs' in row:
            assert value.hexdigest() == row['lfs']['sha256'], name
        else:
            blob = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode() + path.read_bytes()).hexdigest()
            assert blob == row['blobId'], name
        records.append({'file': name, 'sha256': value.hexdigest(), 'bytes': path.stat().st_size,
                        'expected_lfs_sha256': row.get('lfs', {}).get('sha256')})
    receipt = {'repo': 'Qwen/Qwen2.5-VL-7B-Instruct', 'revision': meta['sha'],
         'directory': str(TARGET), 'files': records, 'transport': 'ModelScope direct ranges; frozen HF hash identity',
         'public_mirror_mutable_name_not_version_authority': True, 'external_inference': False}
    with (ROOT / 'vlm_assets.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({'download_complete': True, 'bytes': sum(r['bytes'] for r in records)}))


if __name__ == '__main__':
    main()
