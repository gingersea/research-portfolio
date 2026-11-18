"""Pinned official native depth model/code; public download, no data upload."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import requests

ROOT=Path(__file__).resolve().parent
ASSETS=Path('/data/AI_Research/shared/identity_recovery_20261115')
CODE='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
REV='3bc65d4e14a6786a61acec16453c50e12bf5f338'
SHA='b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fetch(row):
    path=ASSETS/'Depth-Anything-V2'/row['path']
    if path.exists():data=path.read_bytes()
    else:
        for attempt in range(3):
            try:
                r=requests.get(f'https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/{CODE}/{row["path"]}',timeout=30)
                r.raise_for_status();data=r.content;break
            except requests.RequestException:
                if attempt==2:raise
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['sha']
    if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return {'file':row['path'],'sha256':sha(path),'git_blob':row['sha']}


def main():
    r=requests.get(f'https://api.github.com/repos/DepthAnything/Depth-Anything-V2/git/trees/{CODE}?recursive=1',timeout=30)
    r.raise_for_status()
    wanted=[row for row in r.json()['tree'] if row['type']=='blob' and
            (row['path'].startswith('metric_depth/depth_anything_v2/') and row['path'].endswith('.py')
             or row['path'] in ['LICENSE','README.md','metric_depth/README.md','metric_depth/requirements.txt'])]
    sources=list(ThreadPoolExecutor(max_workers=4).map(fetch,wanted))
    (ROOT/'depth_source.json').write_text(json.dumps({'revision':CODE,'files':sources},indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('ranges',ROOT.parent/'physics_methods_20261114/fetch_ranges.py')
    ranges=importlib.util.module_from_spec(spec);spec.loader.exec_module(ranges)
    item=SimpleNamespace(rfilename='depth_anything_v2_metric_hypersim_vits.pth',size=99222290,lfs=SimpleNamespace(sha256=SHA))
    checkpoint=ranges.transfer('depth-anything/Depth-Anything-V2-Metric-Hypersim-Small',REV,item,ASSETS,16)
    assert sha(checkpoint)==SHA
    record={'source_revision':CODE,'source_directory':str(ASSETS/'Depth-Anything-V2'),'checkpoint':str(checkpoint),
            'checkpoint_sha256':SHA,'model_revision':REV,'files':sources}
    (ROOT/'depth_assets.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
