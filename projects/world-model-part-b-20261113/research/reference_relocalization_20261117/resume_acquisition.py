"""Resume identical bounded bytes with higher concurrency; preserve earlier attempts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import time
from urllib.parse import urlparse,urljoin
import requests
from acquire_validation import ROOT,PRIOR,DATA,BLOCK,LIMIT,HOSTS,path_for,inspect

def main():
    out=ROOT/'acquisition_v2';out.mkdir(exist_ok=False)
    old=json.loads((PRIOR/'manifest.json').read_text());excluded={r['participant'] for r in old['cases']}
    meta=json.loads((PRIOR/'archive_metadata/range_probe.json').read_text())
    base=meta['strings'][1].replace('dl=0','dl=1');total=meta['size_fields']['2'];tag=meta['headers']['Etag']
    change={'utc':datetime.now(timezone.utc).isoformat(),'only_change':'36 concurrent range requests instead of12; interrupted partial bytes retained',
            'selection_or_size_budget_changed':False,'new_validation_inference_run':False,
            'old_process_pid':2923725,'transport':'same requests environment, TLS verification and publisher hostname allowlist'}
    (out/'CHANGE.json').write_text(json.dumps(change,indent=2))
    def fetch(i):
        if path_for(i).exists():return {'index':i,'status':'reused'}
        start=i*BLOCK;end=start+BLOCK-1;errors=[]
        for attempt in range(6):
            try:
                with requests.Session() as s:
                    url=base+f'&range_start={start}&range_end={end}&study=reference_v2_20261117&attempt={attempt}'
                    for _ in range(8):
                        host=urlparse(url).hostname;assert any(host==h or host.endswith('.'+h) for h in HOSTS)
                        r=s.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(15,90))
                        if not r.is_redirect:break
                        url=urljoin(url,r.headers['Location']);r.close()
                    assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{total}'
                    assert r.headers['ETag']==tag
                    p=DATA/'parts8'/f'{i:05d}.v2attempt{attempt}.partial';digest=hashlib.sha256()
                    with p.open('xb') as f:
                        for b in r.iter_content(1<<18):f.write(b);digest.update(b)
                    r.close();assert p.stat().st_size==BLOCK
                    p.rename(DATA/'parts8'/f'{i:05d}.part')
                    record={'index':i,'status':'ok','sha256':digest.hexdigest(),'errors':errors};break
            except Exception as e:errors.append(repr(e));time.sleep(1)
        else:record={'index':i,'status':'failed','errors':errors}
        (out/f'{i:05d}.json').write_text(json.dumps(record,indent=2));return record
    begin=time.monotonic()
    while True:
        count,scanned,chosen,boundary=inspect(excluded)
        print(json.dumps({'bytes':count*BLOCK,'scanned':len(scanned),'qualifying_new_participants':len(chosen),
                         'ids':[r['id'] for r in chosen],'seconds':time.monotonic()-begin}),flush=True)
        if len(chosen)>=6 or count==LIMIT:break
        missing=[i for i in range(count,min(count+36,LIMIT)) if not path_for(i).exists()]
        with ThreadPoolExecutor(36) as pool:records=list(pool.map(fetch,missing))
        if any(r['status']=='failed' for r in records):raise RuntimeError('bounded transport retries exhausted; preserve all attempts')
    archive_path=DATA/'ego_points_validation_prefix.tar';digest=hashlib.sha256();ranges=[]
    with archive_path.open('xb') as f:
        for i in range(count):
            b=path_for(i).read_bytes();assert len(b)==BLOCK;digest.update(b);f.write(b)
            ranges.append({'index':i,'path':str(path_for(i)),'sha256':hashlib.sha256(b).hexdigest()})
    record={'archive':str(archive_path),'bytes':count*BLOCK,'sha256':digest.hexdigest(),'whole_archive_bytes':total,'etag':tag,
        'excluded_participants':sorted(excluded),'selected':chosen,'scanned':scanned,'boundary_clip':boundary,
        'full_archive_claim':False,'no_new_prediction_scores_read':True,'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
        'ranges':ranges,'transport_revision':2}
    (ROOT/'selection.json').write_text(json.dumps(record,indent=2));print(json.dumps({'selection_frozen':True,'new_participants':len(chosen)}))

if __name__=='__main__':main()
