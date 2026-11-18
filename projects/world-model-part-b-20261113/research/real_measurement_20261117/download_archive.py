"""Pinned author tar via validated HTTP ranges; no protected/Google services."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse

import requests

from inspect_sources import ROOT, ALLOWED

DATA = Path('/data/AI_Research/shared/real_measurement_20261117')
CHUNK = 32 * 1024 * 1024


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--prefix', action='store_true', help='Only the frozen first3GiB pilot window')
    args=parser.parse_args()
    record=json.loads((ROOT/'archive_metadata/range_probe.json').read_text())
    archive_size=record['size_fields']['2']
    size=min(archive_size,3*1024**3) if args.prefix else archive_size
    url=record['strings'][1].replace('dl=0','dl=1')
    tag=record['headers']['Etag']
    DATA.mkdir(exist_ok=True)
    parts=DATA/'parts'
    parts.mkdir(exist_ok=True)
    started=time.monotonic()
    def fetch(index):
        start=index*CHUNK
        end=min(start+CHUNK,size)-1
        target=parts/f'{index:05d}.part'
        if target.exists():
            assert target.stat().st_size==end-start+1
            return index,target.stat().st_size
        for attempt in range(3):
            try:
                session=requests.Session()
                current=url
                for _ in range(8):
                    host=urlparse(current).hostname
                    assert any(host==h or host.endswith('.'+h) for h in ALLOWED)
                    response=session.get(current, headers={'Range':f'bytes={start}-{end}',
                        'Accept-Encoding':'identity'}, timeout=(20,120), allow_redirects=False)
                    if not response.is_redirect:
                        break
                    current=urljoin(current,response.headers['Location'])
                assert response.status_code==206, response.status_code
                assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{archive_size}', response.headers.get('Content-Range')
                assert response.headers.get('ETag')==tag, response.headers.get('ETag')
                assert len(response.content)==end-start+1
                with target.open('xb') as stream:
                    stream.write(response.content)
                return index,len(response.content)
            except Exception:
                if attempt==2:
                    raise
                time.sleep(1)
    if args.probe:
        item=fetch(0)
        result={'part':item[0],'bytes':item[1],'seconds':time.monotonic()-started,'whole_archive_bytes':size}
        with (ROOT/'archive_metadata/download_probe.json').open('x') as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps(result))
        return
    count=(size+CHUNK-1)//CHUNK
    with ThreadPoolExecutor(8) as pool:
        futures=[pool.submit(fetch,index) for index in range(count)]
        for done,future in enumerate(as_completed(futures),1):
            future.result()
            if done%50==0 or done==count:
                print(json.dumps({'parts':done,'planned':count,'elapsed_seconds':time.monotonic()-started}),flush=True)
    target=DATA/('ego_points_prefix.tar' if args.prefix else 'ego_points.tar')
    digest=hashlib.sha256()
    with target.open('xb') as output:
        for index in range(count):
            with (parts/f'{index:05d}.part').open('rb') as source:
                for block in iter(lambda:source.read(1<<20),b''):
                    digest.update(block)
                    output.write(block)
    assert target.stat().st_size==size
    result={'file':str(target),'sha256':digest.hexdigest(),'bytes':size,'etag':tag,'parts':count,
            'whole_archive_bytes':archive_size,'is_complete_archive':not args.prefix,
            'author_url':record['strings'][1],'elapsed_seconds':time.monotonic()-started,
            'all_range_boundaries_and_etag_verified':True,'author_published_whole_file_checksum_available':False}
    with (ROOT/'archive_metadata'/('prefix_receipt.json' if args.prefix else 'download_receipt.json')).open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    main()
