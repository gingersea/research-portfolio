"""Download public pinned research assets and verify LFS hashes; no API inference."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

STUDY=Path(__file__).resolve().parent
CACHE=Path('/data/AI_Research/shared/physics_methods_20261114/models')
TARGETS={
    'vpt':('zhengzhou/VPT','edacaa9709af842a017361d4140fffac3b74fecc',
           ['pytorch_lora_weights.safetensors']),
    'videophy2':('videophysics/videophy_2_auto','ca38370e1dcc79c99c7017dd8755d0a087299afc',
                 ['*.json','*.model','*.bin','README.md']),
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=TARGETS,required=True)
    args=parser.parse_args();repo,revision,patterns=TARGETS[args.target]
    info=HfApi(token=False).model_info(repo,revision=revision,files_metadata=True)
    assert info.sha==revision and info.gated is False
    destination=CACHE/args.target
    snapshot_download(repo_id=repo,revision=revision,local_dir=destination,
                      allow_patterns=patterns,token=False,max_workers=2)
    rows=[]
    for item in info.siblings:
        path=destination/item.rfilename
        if not path.is_file():continue
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert path.stat().st_size==item.size,path
        expected=item.lfs.sha256 if item.lfs else None
        if expected:assert actual==expected,path
        rows.append({'file':item.rfilename,'bytes':item.size,'sha256':actual,
                     'upstream_lfs_sha256':expected})
    assert rows
    manifest={'repo':repo,'revision':revision,'gated':False,'local_dir':str(destination),
              'verified':True,'files':rows,'method':'public download only; no paid inference/API model call'}
    (STUDY/f'assets_{args.target}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'target':args.target,'verified':True,'files':len(rows),
                      'bytes':sum(row['bytes'] for row in rows)},indent=2))


if __name__=='__main__':main()
