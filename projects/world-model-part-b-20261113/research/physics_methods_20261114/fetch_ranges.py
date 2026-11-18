"""Resumable public HTTP-range transfer with pinned size/blob/LFS verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

from huggingface_hub import HfApi
import requests

STUDY=Path(__file__).resolve().parent
BASE=Path('/data/AI_Research/shared/physics_methods_20261114/models')
REPOS={'vpt':('zhengzhou/VPT','edacaa9709af842a017361d4140fffac3b74fecc'),
       'videophy2':('videophysics/videophy_2_auto','ca38370e1dcc79c99c7017dd8755d0a087299afc')}
LOCAL=threading.local()


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def session():
    if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
    return LOCAL.session


def transfer(repo,revision,item,destination,workers):
    path=destination/item.rfilename;path.parent.mkdir(parents=True,exist_ok=True)
    expected=item.lfs.sha256 if item.lfs else None
    if path.is_file() and path.stat().st_size==item.size and (not expected or sha(path)==expected):return path
    url=f'https://huggingface.co/{repo}/resolve/{revision}/{item.rfilename}'
    chunks=destination/'.parts'/item.rfilename;chunks.mkdir(parents=True,exist_ok=True)
    block=4*1024*1024
    def part(i):
        start=i*block;end=min(item.size-1,start+block-1);size=end-start+1
        target=chunks/str(i)
        if target.is_file() and target.stat().st_size==size:return size
        for attempt in range(3):
            try:
                with session().get(url+f'?download=true&range_part={i}',
                     headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                     stream=True,timeout=(25,150)) as response:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{item.size}':
                        raise ValueError(f'Unexpected range response status{response.status_code}')
                    temp=target.with_suffix('.tmp');count=0
                    with temp.open('wb') as stream:
                        for data in response.iter_content(65536):
                            count+=len(data)
                            if count>size:raise ValueError('Range size exceeded')
                            stream.write(data)
                    if count!=size:raise ValueError('Incomplete range')
                    temp.replace(target);return count
            except Exception as error:
                if attempt==2:raise RuntimeError(f'part{i} failed: {type(error).__name__}') from None
        raise RuntimeError('unreachable')
    number=(item.size+block-1)//block;started=time.monotonic();done=0;received=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task in as_completed([pool.submit(part,i) for i in range(number)]):
            received+=task.result();done+=1
            if done%8==0 or done==number:
                print(json.dumps({'file':item.rfilename,'parts':done,'total_parts':number,
                                  'completed_bytes':received,'seconds':round(time.monotonic()-started,1)}),flush=True)
    assembled=path.with_suffix(path.suffix+'.assembling')
    with assembled.open('wb') as output:
        for i in range(number):
            with (chunks/str(i)).open('rb') as source:shutil.copyfileobj(source,output)
    assert assembled.stat().st_size==item.size
    if expected:assert sha(assembled)==expected,'Upstream LFS hash mismatch'
    else:
        data=assembled.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item.blob_id
    assembled.replace(path);return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=REPOS,required=True)
    parser.add_argument('--workers',type=int,default=32);args=parser.parse_args()
    assert 1<=args.workers<=64
    repo,revision=REPOS[args.target]
    info=HfApi(token=False).model_info(repo,revision=revision,files_metadata=True)
    assert info.sha==revision and info.gated is False
    wanted=[x for x in info.siblings if (args.target=='vpt' and x.rfilename=='pytorch_lora_weights.safetensors')
            or (args.target=='videophy2' and x.rfilename.endswith(('.bin','.json','.model','.md')))]
    destination=BASE/(args.target+'_ranged');rows=[]
    for item in wanted:
        path=transfer(repo,revision,item,destination,args.workers)
        rows.append({'file':item.rfilename,'bytes':item.size,'sha256':sha(path),
                     'upstream_lfs_sha256':item.lfs.sha256 if item.lfs else None,'upstream_git_blob':item.blob_id})
    manifest={'repo':repo,'revision':revision,'local_dir':str(destination),'verified':True,
              'files':rows,'transport':'public HTTP ranges; original pinned file bytes, no model/API inference'}
    (STUDY/f'assets_{args.target}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'verified':True,'target':args.target,'bytes':sum(x['bytes'] for x in rows)},indent=2),flush=True)


if __name__=='__main__':main()
