"""Fetch only runnable official text blobs, verified against pinned Git trees."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import requests

BASE=Path('/data/AI_Research/shared/physics_methods_20261114')
LOCAL=threading.local()
REPOS={
 'vpt':('Tom-zgt/VPT','VPT_code',['finetrainers','examples','setup.py','train.py','requirements.txt','README.md','LICENSE']),
 'videophy':('Hritikbansal/videophy','videophy_code',['VIDEOPHY2/mplug_owl_video','VIDEOPHY2/inference.py','VIDEOPHY2/template.py','VIDEOPHY2/utils.py','VIDEOPHY2/data_utils','VIDEOPHY2/examples','README.md','requirements.txt','LICENSE'])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',choices=REPOS,required=True)
    args=parser.parse_args();repo,clone,paths=REPOS[args.repo];tree=BASE/'upstream'/clone
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
    # Asking ls-tree for blob sizes (-l) defeats a blobless checkout by fetching
    # every missing blob. The Git object ID already verifies the complete bytes.
    records=subprocess.check_output(['git','ls-tree','-r',revision,*paths],cwd=tree,text=True).splitlines()
    files=[]
    for record in records:
        meta,name=record.split('\t',1);parts=meta.split()
        if Path(name).suffix in {'.py','.json','.sh','.txt','.md','.yaml','.yml','.toml','.csv'} or Path(name).name=='LICENSE':
            files.append((name,parts[2]))
    out=BASE/'source'/args.repo
    def fetch(item):
        name,expected=item;path=out/name
        def check(data):
            return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==expected
        if path.is_file() and check(path.read_bytes()):return name
        if not hasattr(LOCAL,'http'):LOCAL.http=requests.Session()
        for attempt in range(3):
            try:
                r=LOCAL.http.get(f'https://raw.githubusercontent.com/{repo}/{revision}/{name}',timeout=(20,100))
                if r.status_code!=200 or not check(r.content):raise ValueError('Blob not verified')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content);return name
            except Exception as error:
                if attempt==2:raise RuntimeError(f'{name}: {type(error).__name__}') from None
    completed=0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for result in as_completed([pool.submit(fetch,item) for item in files]):
            result.result();completed+=1
            if completed%20==0 or completed==len(files):
                print(json.dumps({'repo':repo,'files':completed,'total':len(files)}),flush=True)
    manifest={'repo':repo,'revision':revision,'local_dir':str(out),'files':[
        {'path':name,'git_blob':blob,'bytes':(out/name).stat().st_size} for name,blob in files],'verified':True}
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'repo':repo,'revision':revision,'verified_files':completed}),flush=True)


if __name__=='__main__':main()
