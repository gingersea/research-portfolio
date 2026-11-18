"""Pinned official DAVIS release via resumable checked byte ranges."""
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import socket
import threading
import time
import zipfile

import requests

ROOT=Path(__file__).resolve().parent
ASSETS=Path('/data/AI_Research/shared/paper_benchmark_20261115')
REV='ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b'
SIZE=1668710491
MD5='a7e786a1a5d26ba941b7c10bed21c20e'
URL='https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip?generation=1671476530685857'
LOCAL=threading.local()


def session():
    if not hasattr(LOCAL,'http'):
        LOCAL.http=requests.Session();LOCAL.http.trust_env=False
    return LOCAL.http


def sha(path,algorithm='sha256'):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()


def part(i):
    block=4*1024*1024;start=i*block;end=min(SIZE-1,start+block-1)
    target=ASSETS/'parts'/str(i);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==end-start+1:return
    for attempt in range(4):
        try:
            r=session().get(URL+f'&range_part={i}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(20,120))
            r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
            assert len(r.content)==end-start+1
            temporary=target.with_suffix('.partial');temporary.write_bytes(r.content);temporary.replace(target);return
        except (requests.RequestException,AssertionError):
            if attempt==3:raise


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    source_url=f'https://raw.githubusercontent.com/google-deepmind/tapnet/{REV}/tapnet/tapvid/evaluation_datasets.py'
    response=requests.get(source_url,timeout=30);response.raise_for_status();source=response.content
    assert hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()=='b81988156c1c5ac2d7f2c699bdf80f67f8ea9028'
    original=ROOT/'official_evaluation_datasets.py'
    if original.exists():assert original.read_bytes()==source
    else:original.write_bytes(source)
    parsed=ast.parse(source.decode());names=['compute_tapvid_metrics','sample_queries_first']
    bodies=[ast.get_source_segment(source.decode(),node) for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(bodies)==2
    portable=ROOT/'official_numpy.py';text='"""Unmodified official metric/query function bodies; dependency-only extraction."""\nfrom typing import Mapping\nimport numpy as np\n\n'+'\n\n'.join(bodies)+'\n'
    if portable.exists():assert portable.read_text()==text
    else:portable.write_text(text)
    source_record={'revision':REV,'source_url':source_url,'original_sha256':sha(original),'numpy_extraction_sha256':sha(portable),
                   'functions':names,'function_bodies_modified':False}
    (ROOT/'official_source.json').write_text(json.dumps(source_record,indent=2)+'\n')
    # Direct access is available, but local DNS failed for this public hostname.
    # Resolve its public A record via DoH; keep the original HTTPS host/SNI and
    # normal certificate verification. No URL, permission or data bytes change.
    dns=requests.get('https://dns.google/resolve?name=storage.googleapis.com&type=A',timeout=30)
    dns.raise_for_status();addresses=[r['data'] for r in dns.json()['Answer'] if r['type']==1]
    assert addresses
    native_resolve=socket.getaddrinfo
    public_address=addresses[0]
    socket.getaddrinfo=lambda host,port,*a,**kw:native_resolve(public_address if host=='storage.googleapis.com' else host,port,*a,**kw)
    print(json.dumps({'transport':'direct verified HTTPS; public DNS fallback','public_A':public_address}),flush=True)
    archive=ASSETS/'tapvid_davis.zip'
    if not archive.exists():
        started=time.monotonic();number=(SIZE+4*1024*1024-1)//(4*1024*1024)
        with ThreadPoolExecutor(max_workers=24) as executor:
            jobs=[executor.submit(part,i) for i in range(number)]
            for done,job in enumerate(as_completed(jobs),1):
                job.result()
                if done%24==0 or done==number:print(json.dumps({'parts':done,'total':number,'seconds':round(time.monotonic()-started,1)}),flush=True)
        temporary=archive.with_suffix('.assembling')
        with temporary.open('wb') as out:
            for i in range(number):
                with (ASSETS/'parts'/str(i)).open('rb') as inp:shutil.copyfileobj(inp,out)
        assert temporary.stat().st_size==SIZE and sha(temporary,'md5')==MD5;temporary.replace(archive)
    assert archive.stat().st_size==SIZE and sha(archive,'md5')==MD5
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        for info in entries:
            relative=Path(info.filename);assert not relative.is_absolute() and '..' not in relative.parts
            if not info.is_dir():
                target=ASSETS/'release'/relative
                if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
        files=[{'file':i.filename,'bytes':i.file_size,'sha256':sha(ASSETS/'release'/i.filename)} for i in entries if not i.is_dir()]
    record={'official_url':URL,'generation':'1671476530685857','zip':str(archive),'bytes':SIZE,'md5':MD5,
            'sha256':sha(archive),'release_directory':str(ASSETS/'release'),'files':files,'official_source':source_record}
    (ROOT/'dataset_assets.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
