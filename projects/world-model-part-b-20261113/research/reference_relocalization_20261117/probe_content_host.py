"""Check the publisher's direct-content hostname without changing infrastructure."""
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin,urlparse
import requests

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    meta=json.loads((ROOT.parent/'real_measurement_20261117/archive_metadata/range_probe.json').read_text())
    url=meta['strings'][1].replace('www.dropbox.com','dl.dropboxusercontent.com').replace('dl=0','dl=1')+'&probe=content_20261117'
    record={'direct_request_only':True,'global_environment_changed':False,'steps':[]}
    start=time.monotonic()
    try:
        with requests.Session() as s:
            s.trust_env=False
            for _ in range(8):
                host=urlparse(url).hostname
                assert any(host==h or host.endswith('.'+h) for h in ['dropbox.com','dropboxusercontent.com'])
                r=s.get(url,headers={'Range':'bytes=0-1048575','Accept-Encoding':'identity'},stream=True,allow_redirects=False,timeout=(12,20))
                record['steps'].append({'host':host,'status':r.status_code})
                if not r.is_redirect:break
                url=urljoin(url,r.headers['Location']);r.close()
            record.update(status=r.status_code,headers=dict(r.headers))
            assert r.status_code==206 and r.headers['Content-Range']==f"bytes 0-1048575/{meta['size_fields']['2']}"
            body=r.content;r.close()
            record.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    except Exception as error:
        record['error']=repr(error)
    record['elapsed_seconds']=time.monotonic()-start
    (ROOT/'content_transport_probe.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
