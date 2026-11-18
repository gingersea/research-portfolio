"""Resume fixed range bytes after resets, using1MiB requests and8 workers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import time
from urllib.parse import urlparse,urljoin
import requests
from acquire_validation import ROOT,PRIOR,DATA,BLOCK,LIMIT,HOSTS,path_for,inspect

def main():
    out=ROOT/'acquisition_v3';out.mkdir(exist_ok=False)
    meta=json.loads((PRIOR/'archive_metadata/range_probe.json').read_text());total=meta['size_fields']['2'];tag=meta['headers']['Etag']
    base=meta['strings'][1].replace('dl=0','dl=1')
    excluded={r['participant'] for r in json.loads((PRIOR/'manifest.json').read_text())['cases']}
    (out/'CHANGE.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
       'reason':'36-way attempt caused repeated connection resets; preserve all failures and switch to8-way1MiB resumable requests',
       'same_bytes_size_budget_selection':True,'no_new_validation_predictions':True},indent=2))
    pieces=DATA/'pieces1';pieces.mkdir(exist_ok=False)
    def fetch(i):
        if path_for(i).exists():return
        records=[]
        for slot in range(8):
            begin=i*BLOCK+slot*(1<<20);end=begin+(1<<20)-1
            file=pieces/f'{i:05d}_{slot}.part';errors=[]
            for attempt in range(6):
                try:
                    with requests.Session() as s:
                        url=base+f'&range_start={begin}&range_end={end}&study=reference_v3_20261117&attempt={attempt}'
                        for _ in range(8):
                            host=urlparse(url).hostname;assert any(host==h or host.endswith('.'+h) for h in HOSTS)
                            r=s.get(url,headers={'Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(15,60))
                            if not r.is_redirect:break
                            url=urljoin(url,r.headers['Location']);r.close()
                        assert r.status_code==206 and r.headers['Content-Range']==f'bytes {begin}-{end}/{total}'
                        assert r.headers['ETag']==tag
                        body=r.content;r.close();assert len(body)==1<<20
                        with file.open('xb') as f:f.write(body)
                        records.append({'slot':slot,'sha256':hashlib.sha256(body).hexdigest(),'errors':errors})
                        break
                except Exception as error:errors.append(repr(error));time.sleep(2)
            else:
                (out/f'{i:05d}_failure.json').write_text(json.dumps({'slot':slot,'errors':errors},indent=2))
                raise RuntimeError((i,slot,'bounded requests failed'))
        path=DATA/'parts8'/f'{i:05d}.part';digest=hashlib.sha256()
        with path.open('xb') as f:
            for slot in range(8):b=(pieces/f'{i:05d}_{slot}.part').read_bytes();f.write(b);digest.update(b)
        (out/f'{i:05d}.json').write_text(json.dumps({'index':i,'status':'ok','sha256':digest.hexdigest(),'pieces':records},indent=2))
        print(json.dumps({'completed_range':i}),flush=True)
    start=time.monotonic()
    while True:
        count,scanned,chosen,boundary=inspect(excluded)
        print(json.dumps({'bytes':count*BLOCK,'scanned':len(scanned),'qualifying_new_participants':len(chosen),
            'ids':[r['id'] for r in chosen],'seconds':time.monotonic()-start}),flush=True)
        if len(chosen)>=6 or count==LIMIT:break
        missing=[i for i in range(count,min(count+8,LIMIT)) if not path_for(i).exists()]
        with ThreadPoolExecutor(8) as pool:list(pool.map(fetch,missing))
    dest=DATA/'ego_points_validation_prefix.tar';digest=hashlib.sha256();ranges=[]
    with dest.open('xb') as f:
        for i in range(count):
            b=path_for(i).read_bytes();assert len(b)==BLOCK;digest.update(b);f.write(b)
            ranges.append({'index':i,'path':str(path_for(i)),'sha256':hashlib.sha256(b).hexdigest()})
    record={'archive':str(dest),'bytes':count*BLOCK,'sha256':digest.hexdigest(),'whole_archive_bytes':total,'etag':tag,
       'excluded_participants':sorted(excluded),'selected':chosen,'scanned':scanned,'boundary_clip':boundary,
       'full_archive_claim':False,'no_new_prediction_scores_read':True,'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
       'ranges':ranges,'transport_revision':3}
    (ROOT/'selection.json').write_text(json.dumps(record,indent=2));print(json.dumps({'selection_frozen':True,'new_participants':len(chosen)}))

if __name__=='__main__':main()
