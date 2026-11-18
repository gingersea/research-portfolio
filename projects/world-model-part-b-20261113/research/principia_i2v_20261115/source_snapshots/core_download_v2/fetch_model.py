"""Fetch pinned native LTX I2V components; text encoder reuse is recorded separately."""
import argparse
import importlib.util
import json
from pathlib import Path
import threading

import requests

from huggingface_hub import HfApi

STUDY = Path(__file__).resolve().parent
REPO = 'Lightricks/LTX-Video-0.9.5'
REVISION = 'e58e28c39631af4d1468ee57a853764e11c1d37e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--transport', choices=['hf_proxy', 'modelscope_direct'], default='modelscope_direct')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('range_transfer', STUDY.parent / 'physics_methods_20261114/fetch_ranges.py')
    transfer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfer)
    if args.transport == 'modelscope_direct':
        local = threading.local()
        class MirrorHTTP:
            def __init__(self):
                self.direct = requests.Session()
                self.direct.trust_env = False
                self.proxy = requests.Session()
            def get(self, url, **kwargs):
                prefix = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'
                if url.startswith(prefix) and url.split('?', 1)[0].endswith('.safetensors'):
                    mirrored = url.replace(prefix, f'https://modelscope.cn/models/{REPO}/resolve/master/', 1)
                    return self.direct.get(mirrored, **kwargs)
                return self.proxy.get(url, **kwargs)
        def mirror_session():
            if not hasattr(local, 'http'):
                local.http = MirrorHTTP()
            return local.http
        transfer.session = mirror_session
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION and not info.gated
    wanted = [item for item in info.siblings if item.rfilename.startswith(('transformer/', 'vae/', 'tokenizer/', 'scheduler/')) or item.rfilename in ['model_index.json', 'README.md']]
    wanted.sort(key=lambda item: (item.size > 10_000_000, item.rfilename))
    rows = []
    for item in wanted:
        path = transfer.transfer(REPO, REVISION, item, args.destination, args.workers)
        if not item.lfs:
            import hashlib
            data = path.read_bytes()
            assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item.blob_id
        rows.append({'file': item.rfilename, 'bytes': item.size, 'sha256': transfer.sha(path), 'lfs_sha256': item.lfs.sha256 if item.lfs else None, 'git_blob': item.blob_id})
    manifest = {'repo': REPO, 'revision': REVISION, 'local_dir': str(args.destination), 'transport': args.transport, 'mirror_integrity': 'Mirror weight bytes must equal the pinned HF LFS SHA256; mutable mirror name is not trusted as model identity.', 'files': rows, 'scope': 'Pinned native transformer, VAE, tokenizer and scheduler. Text encoder supplied separately from existing T5-v1.1-XXL cache; no T2V-to-I2V patch.'}
    target = STUDY / 'native_assets.json'
    if target.exists():
        raise FileExistsError('Refuse to replace manifest')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'verified_bytes': sum(row['bytes'] for row in rows), 'files': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
