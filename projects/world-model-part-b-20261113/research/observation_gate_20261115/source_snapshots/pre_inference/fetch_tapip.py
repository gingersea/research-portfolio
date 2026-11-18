"""Pinned TAPIP3D native source/checkpoint and extension source, no installation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import requests

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/data/AI_Research/shared/observation_gate_20261115')
CODE = '4cb7e69a1687f67d56ec3e506768f51f2c581b46'
MODEL = '08730a588204f258f7a86530df15b2a6426e5f5b'
CHECKSUM = '3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def fetch(row):
    relative, blob = row['path'], row['sha']
    target = ASSETS/'TAPIP3D'/relative
    if target.exists():
        data=target.read_bytes()
    else:
        for attempt in range(3):
            try:
                response=requests.get(f'https://raw.githubusercontent.com/zbw001/TAPIP3D/{CODE}/{relative}',timeout=30)
                response.raise_for_status()
                data=response.content
                break
            except requests.RequestException:
                if attempt==2:raise
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert target.read_bytes()==data
    else:
        target.write_bytes(data)
    return {'file':relative,'git_blob':blob,'sha256':sha(target)}


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    response=requests.get(f'https://api.github.com/repos/zbw001/TAPIP3D/git/trees/{CODE}?recursive=1',timeout=30)
    response.raise_for_status()
    tree=response.json()['tree']
    rows=[row for row in tree if row['type']=='blob' and
          (row['path'].startswith(('models/','utils/','configs/','datasets/','training/','third_party/cotracker/','third_party/pointops2/'))
           and row['path'].endswith(('.py','.yaml','.cu','.cpp','.h','.hpp'))
           or row['path'] in ['README.md','LICENSE','requirements.txt'])]
    sources=list(ThreadPoolExecutor(max_workers=4).map(fetch,rows))
    (ROOT/'tapip_source.json').write_text(json.dumps({'revision':CODE,'directory':str(ASSETS/'TAPIP3D'),'files':sources},indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('range_transfer',ROOT.parent/'physics_methods_20261114/fetch_ranges.py')
    transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
    item=SimpleNamespace(rfilename='tapip3d_final.pth',size=309417450,lfs=SimpleNamespace(sha256=CHECKSUM))
    weight=transfer.transfer('zbww/tapip3d',MODEL,item,ASSETS,16)
    assert sha(weight)==CHECKSUM
    record={'source_revision':CODE,'source_directory':str(ASSETS/'TAPIP3D'),'model_repo':'zbww/tapip3d',
            'model_revision':MODEL,'checkpoint':str(weight),'checkpoint_sha256':CHECKSUM,'source_files':len(sources)}
    (ROOT/'tapip_assets.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
