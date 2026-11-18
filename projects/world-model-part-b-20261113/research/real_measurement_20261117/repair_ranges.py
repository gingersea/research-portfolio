"""Repair missing original ranges only; preserve all first transport attempts."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time
from urllib.parse import urljoin,urlparse

import requests

from inspect_sources import ROOT,ALLOWED

DATA=Path('/data/AI_Research/shared/real_measurement_20261117')
BLOCK=8*1024**2


if __name__=='__main__':
    folder=DATA/'parts8'
    metadata=json.loads((ROOT/'archive_metadata/range_probe.json').read_text())
    url=metadata['strings'][1].replace('dl=0','dl=1')
    size=metadata['size_fields']['2']
    tag=metadata['headers']['Etag']
    top=max(int(p.stem) for p in folder.glob('*.part'))
    missing=[i for i in range(top+1) if not (folder/f'{i:05d}.part').exists()]
    output=ROOT/'range_repair'
    output.mkdir(exist_ok=False)
    (output/'plan.json').write_text(json.dumps({'missing':missing,'highest_completed_before_repair':top,
        'selection_change':False,'original_failed_receipts_preserved':True},indent=2))
    print(json.dumps({'missing':missing,'highest':top}),flush=True)
    def one(index):
        start=index*BLOCK
        end=start+BLOCK-1
        errors=[]
        for attempt in range(4):
            temporary=folder/f'{index:05d}.repair{attempt}.partial'
            try:
                session=requests.Session()
                current=url+f'&range_start={start}&range_end={end}&transport_retry={attempt}'
                for _ in range(8):
                    host=urlparse(current).hostname
                    assert any(host==h or host.endswith('.'+h) for h in ALLOWED)
                    response=session.get(current,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                       timeout=(15,60),allow_redirects=False,stream=True)
                    if not response.is_redirect:
                        break
                    current=urljoin(current,response.headers['Location'])
                    response.close()
                assert response.status_code==206
                assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{size}'
                assert response.headers.get('ETag')==tag
                with temporary.open('xb') as stream:
                    for chunk in response.iter_content(1<<18):
                        stream.write(chunk)
                response.close()
                assert temporary.stat().st_size==BLOCK
                target=folder/f'{index:05d}.part'
                assert not target.exists()
                temporary.replace(target)
                record={'index':index,'status':'ok','errors':errors,'original_range':f'{start}-{end}'}
                (output/f'{index:05d}.json').write_text(json.dumps(record,indent=2))
                return record
            except Exception as error:
                errors.append(repr(error))
                time.sleep(1)
        record={'index':index,'status':'failed','errors':errors}
        (output/f'{index:05d}.json').write_text(json.dumps(record,indent=2))
        return record
    with ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(one,i) for i in missing]
        records=[f.result() for f in as_completed(futures)]
    (output/'receipt.json').write_text(json.dumps({'cases':records,'all_ok':all(r['status']=='ok' for r in records)},indent=2))
    print(json.dumps({'repaired':sum(r['status']=='ok' for r in records),'planned':len(records)}))
