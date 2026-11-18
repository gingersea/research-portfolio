"""Bounded prefix acquisition with streamed receipts and unique original URLs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin,urlparse

import requests

from inspect_sources import ROOT,ALLOWED

DATA=Path('/data/AI_Research/shared/real_measurement_20261117')
BLOCK=8*1024**2
LIMIT=3*1024**3


if __name__=='__main__':
    metadata=json.loads((ROOT/'archive_metadata/range_probe.json').read_text())
    url=metadata['strings'][1].replace('dl=0','dl=1')
    total=metadata['size_fields']['2']
    tag=metadata['headers']['Etag']
    folder=DATA/'parts8'
    folder.mkdir(exist_ok=False)
    old=(DATA/'parts/00000.part').read_bytes()
    for i in range(len(old)//BLOCK):
        (folder/f'{i:05d}.part').write_bytes(old[i*BLOCK:(i+1)*BLOCK])
    start_time=time.monotonic()
    def fetch(index):
        start=index*BLOCK
        end=min(start+BLOCK,LIMIT)-1
        path=folder/f'{index:05d}.part'
        if path.exists():
            return index
        errors=[]
        for attempt in range(2):
            temporary=folder/f'{index:05d}.attempt{attempt}.partial'
            try:
                session=requests.Session()
                current=url+f'&range_start={start}&range_end={end}'
                for _ in range(8):
                    host=urlparse(current).hostname
                    assert any(host==h or host.endswith('.'+h) for h in ALLOWED)
                    response=session.get(current,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                         timeout=(15,60),allow_redirects=False,stream=True)
                    if not response.is_redirect:
                        break
                    current=urljoin(current,response.headers['Location'])
                    response.close()
                assert response.status_code==206,response.status_code
                assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}',dict(response.headers)
                assert response.headers.get('ETag')==tag
                with temporary.open('xb') as stream:
                    for chunk in response.iter_content(1<<18):
                        stream.write(chunk)
                response.close()
                assert temporary.stat().st_size==end-start+1
                temporary.replace(path)
                if errors:
                    (folder/f'{index:05d}.failures.json').write_text(json.dumps(errors))
                return index
            except Exception as error:
                errors.append(repr(error))
        (folder/f'{index:05d}.failures.json').write_text(json.dumps(errors))
        raise RuntimeError((index,errors))
    with ThreadPoolExecutor(12) as pool:
        futures=[pool.submit(fetch,i) for i in range(LIMIT//BLOCK)]
        for completed,future in enumerate(as_completed(futures),1):
            future.result()
            if completed%12==0:
                print(json.dumps({'parts':completed,'total':LIMIT//BLOCK,'seconds':time.monotonic()-start_time}),flush=True)
    target=DATA/'ego_points_prefix.tar'
    digest=hashlib.sha256()
    with target.open('xb') as stream:
        for i in range(LIMIT//BLOCK):
            block=(folder/f'{i:05d}.part').read_bytes()
            digest.update(block)
            stream.write(block)
    receipt={'file':str(target),'sha256':digest.hexdigest(),'bytes':LIMIT,'whole_archive_bytes':total,'etag':tag,
             'author_url':url,'is_complete_archive':False,'all_range_boundaries_and_etag_verified':True,
             'elapsed_seconds':time.monotonic()-start_time,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'archive_preview_415_and_detached_link_403_attempts_retained':True}
    with (ROOT/'archive_metadata/prefix_receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))
