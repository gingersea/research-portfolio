"""Three prespecified official PStudio minival examples, exact GCS object versions."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests

from fetch import sha

ROOT=Path(__file__).resolve().parent
ASSETS=Path('/data/AI_Research/shared/paper_benchmark_20261115/pstudio_pilot')
NAMES=['basketball_5.npz','softball_25.npz','boxes_22.npz']


def fetch(name):
    base='https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/minival_v1.0/'+name
    for attempt in range(4):
        try:
            head=requests.head(base,timeout=(15,30));head.raise_for_status();break
        except requests.RequestException as error:
            print(json.dumps({'event':'metadata_retry','name':name,'attempt':attempt+1,'error':repr(error)}),flush=True)
            if attempt==3:raise
    size=int(head.headers['Content-Length']);generation=head.headers['x-goog-generation']
    encoded=next(v[4:] for v in head.headers['x-goog-hash'].split(', ') if v.startswith('md5='))
    checksum=base64.b64decode(encoded).hex();target=ASSETS/name
    row={'name':name,'official_url':base,'generation':generation,'bytes':size,'md5':checksum}
    (ASSETS/(name+'.upstream.json')).write_text(json.dumps(row,indent=2)+'\n')
    block=1024*1024;parts=ASSETS/(name+'.parts');parts.mkdir(exist_ok=True)

    def part(i):
        start=i*block;end=min(size-1,start+block-1);path=parts/str(i)
        if path.exists() and path.stat().st_size==end-start+1:return
        for attempt in range(3):
            try:
                r=requests.get(base+f'?generation={generation}&pilot_part={i}',headers={'Range':f'bytes={start}-{end}'},timeout=(20,120))
                r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{size}'
                assert len(r.content)==end-start+1;path.write_bytes(r.content);return
            except (requests.RequestException,AssertionError):
                if attempt==2:raise

    if not target.exists():
        list(ThreadPoolExecutor(max_workers=8).map(part,range((size+block-1)//block)))
        with target.open('wb') as f:
            for i in range((size+block-1)//block):f.write((parts/str(i)).read_bytes())
    assert target.stat().st_size==size and sha(target,'md5')==checksum
    row.update(local=str(target),sha256=sha(target),annotation_source='calibrated PStudio geometry, not manual depth labels')
    print(json.dumps({'event':'verified',**row}),flush=True);return row


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    rows=list(ThreadPoolExecutor(max_workers=3).map(fetch,NAMES))
    (ROOT/'pilot3d_assets.json').write_text(json.dumps({'cases':rows,'selection_source':
        'https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapvid3d/colab/load_and_visualize_tapvid3d_samples.ipynb',
        'selection':'first3 PStudio entries in official notebook, frozen in ANALYSIS_PLAN.md before download'},indent=2)+'\n')


if __name__=='__main__':main()
