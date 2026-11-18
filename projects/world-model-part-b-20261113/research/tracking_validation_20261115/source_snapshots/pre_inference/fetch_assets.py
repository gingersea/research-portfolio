"""Pinned public CoTracker3 source and checkpoint, without installing packages."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/data/AI_Research/shared/tracking_validation_20261115')
CODE_REV = '82e02e8029753ad4ef13cf06be7f4fc5facdda4d'
MODEL_REV = 'bf55ea50d4390e1820a267f131cd6587240fb2c5'
WEIGHT_SHA = '2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_source(item):
    relative, git_blob = item
    url = f'https://raw.githubusercontent.com/facebookresearch/co-tracker/{CODE_REV}/{relative}'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content = response.content
    digest = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    assert digest == git_blob, relative
    target = ASSETS / 'co-tracker' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == content
    else:
        target.write_bytes(content)
    return {'file': relative, 'sha256': sha(target), 'git_blob': git_blob}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://api.github.com/repos/facebookresearch/co-tracker/git/trees/{CODE_REV}?recursive=1', timeout=30)
    response.raise_for_status()
    items = [(row['path'], row['sha']) for row in response.json()['tree']
             if row['type'] == 'blob' and (row['path'].startswith('cotracker/') and row['path'].endswith('.py')
                or row['path'] in ['hubconf.py', 'LICENSE', 'LICENSE.md', 'README.md'])]
    records = list(ThreadPoolExecutor(max_workers=8).map(download_source, items))
    weight = ASSETS / 'scaled_offline.pth'
    if not weight.exists():
        url = f'https://huggingface.co/facebook/cotracker3/resolve/{MODEL_REV}/scaled_offline.pth'
        with requests.get(url, stream=True, timeout=(15, 30)) as result:
            result.raise_for_status()
            with (ASSETS / 'scaled_offline.pth.part').open('wb') as stream:
                for chunk in result.iter_content(1024*1024):
                    stream.write(chunk)
        assert sha(ASSETS / 'scaled_offline.pth.part') == WEIGHT_SHA
        (ASSETS / 'scaled_offline.pth.part').rename(weight)
    assert sha(weight) == WEIGHT_SHA and weight.stat().st_size == 101890938
    manifest = {'source_repo': 'facebookresearch/co-tracker', 'source_revision': CODE_REV,
                'model_repo': 'facebook/cotracker3', 'model_revision': MODEL_REV,
                'checkpoint': str(weight), 'checkpoint_sha256': WEIGHT_SHA,
                'source_directory': str(ASSETS / 'co-tracker'), 'source_files': records,
                'no_package_installation': True}
    target = ROOT / 'assets.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'source_files'}, indent=2))


if __name__ == '__main__':
    main()
