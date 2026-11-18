"""Acquire the original public MASt3R weights with verified byte ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parent
DEST=Path('/data/AI_Research/shared/reference_relocalization_20261117/weights')
URL='https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
SIZE=2754910614
ETAG='"a4349996-61ce5970b7b3e"'
BLOCK=16*1024*1024

def get(i):
    start=i*BLOCK
    end=min(start+BLOCK,SIZE)-1
    target=DEST/f'{i:04d}.part'
    failures=[]
    for attempt in range(6):
        path=DEST/f'{i:04d}.attempt{attempt}.partial'
        try:
            with requests.Session() as session:
                session.trust_env=False
                with session.get(URL,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,120),allow_redirects=False) as r:
                    assert r.status_code==206,(r.status_code,r.headers)
                    assert r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
                    assert r.headers['ETag']==ETAG
                    with path.open('xb') as f:
                        for b in r.iter_content(1024*1024):
                            f.write(b)
            assert path.stat().st_size==end-start+1
            path.rename(target)
            return {'index':i,'start':start,'end':end,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'failures':failures}
        except Exception as e:
            failures.append(repr(e))
            time.sleep(2)
    raise RuntimeError((i,failures))

def main():
    DEST.mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=[]
        for r in pool.map(get,range((SIZE+BLOCK-1)//BLOCK)):
            records.append(r)
            print(json.dumps({'complete':len(records),'bytes':sum(x['end']-x['start']+1 for x in records)}),flush=True)
    output=DEST/'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    digest=hashlib.sha256()
    with output.open('xb') as f:
        for i in range(len(records)):
            with (DEST/f'{i:04d}.part').open('rb') as source:
                while b:=source.read(1024*1024):
                    f.write(b)
                    digest.update(b)
    assert output.stat().st_size==SIZE
    receipt={'url':URL,'etag':ETAG,'bytes':SIZE,'checkpoint':str(output),'sha256':digest.hexdigest(),
             'publisher_checksum_available':False,'ranges':records,'license':'CC-BY-NC-SA-4.0; research use only'}
    (ROOT/'checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':
    main()
