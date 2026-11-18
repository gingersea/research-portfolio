"""Read-only direct transport probe for the already authorized public asset."""
import json
from pathlib import Path
import time
from urllib.parse import urlparse,urljoin
import requests

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    prior=ROOT.parent/'real_measurement_20261117/archive_metadata/range_probe.json'
    meta=json.loads(prior.read_text());url=meta['strings'][1].replace('dl=0','dl=1')+'&probe=reference_direct_20261117'
    record={'direct_request_only':True,'global_environment_changed':False}
    started=time.monotonic()
    try:
        with requests.Session() as s:
            s.trust_env=False
            for _ in range(8):
                host=urlparse(url).hostname
                assert any(host==h or host.endswith('.'+h) for h in ['dropbox.com','dropboxusercontent.com'])
                r=s.get(url,headers={'Range':'bytes=0-1048575','Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(12,20))
                if not r.is_redirect:break
                url=urljoin(url,r.headers['Location']);r.close()
            record.update(status=r.status_code,headers=dict(r.headers))
            assert r.status_code==206
            size=sum(len(b) for b in r.iter_content(1<<18));r.close()
            record['bytes']=size
    except Exception as error:
        record['error']=repr(error)
    record['elapsed_seconds']=time.monotonic()-started
    (ROOT/'direct_transport_probe.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
