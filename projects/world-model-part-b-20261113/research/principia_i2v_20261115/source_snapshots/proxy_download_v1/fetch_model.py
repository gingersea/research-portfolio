"""Fetch pinned native LTX I2V components; text encoder reuse is recorded separately."""
import argparse
import importlib.util
import json
from pathlib import Path

from huggingface_hub import HfApi

STUDY = Path(__file__).resolve().parent
REPO = 'Lightricks/LTX-Video-0.9.5'
REVISION = 'e58e28c39631af4d1468ee57a853764e11c1d37e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('range_transfer', STUDY.parent / 'physics_methods_20261114/fetch_ranges.py')
    transfer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfer)
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION and not info.gated
    wanted = [item for item in info.siblings if item.rfilename.startswith(('transformer/', 'vae/', 'tokenizer/', 'scheduler/')) or item.rfilename in ['model_index.json', 'README.md']]
    wanted.sort(key=lambda item: (item.size > 10_000_000, item.rfilename))
    rows = []
    for item in wanted:
        path = transfer.transfer(REPO, REVISION, item, args.destination, args.workers)
        rows.append({'file': item.rfilename, 'bytes': item.size, 'sha256': transfer.sha(path), 'lfs_sha256': item.lfs.sha256 if item.lfs else None, 'git_blob': item.blob_id})
    manifest = {'repo': REPO, 'revision': REVISION, 'local_dir': str(args.destination), 'files': rows, 'scope': 'Pinned native transformer, VAE, tokenizer and scheduler. Text encoder supplied separately from existing T5-v1.1-XXL cache; no T2V-to-I2V patch.'}
    target = STUDY / 'native_assets.json'
    if target.exists():
        raise FileExistsError('Refuse to replace manifest')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'verified_bytes': sum(row['bytes'] for row in rows), 'files': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
