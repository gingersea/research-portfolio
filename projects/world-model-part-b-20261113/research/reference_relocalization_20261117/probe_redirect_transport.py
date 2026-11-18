"""One bounded direct check on the publisher-issued download redirect."""
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin,urlparse
import requests
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    meta=json.loads((ROOT.parent/'real_measurement_20261117/archive_metadata/range_probe.json').read_text())
    url=meta['strings'][1].replace('dl=0','dl=1')+'&probe=publisher_redirect_20261117'
    record={'infrastructure_changed':False,'attempts':1};start=time.monotonic()
    try:
        with requests.Session() as proxy:
            for _ in range(8):
                host=urlparse(url).hostname;assert any(host==h or host.endswith('.'+h) for h in ['dropbox.com','dropboxusercontent.com'])
                r=proxy.get(url,headers={'Range':'bytes=0-1048575'},stream=True,allow_redirects=False,timeout=(12,20))
                if not r.is_redirect:break
                url=urljoin(url,r.headers['Location']);r.close()
            assert r.status_code==206;r.close()
        assert urlparse(url).hostname.endswith('.dropboxusercontent.com')
        download_start=time.monotonic()
        with requests.Session() as direct:
            direct.trust_env=False
            r=direct.get(url,headers={'Range':'bytes=0-1048575'},stream=True,allow_redirects=False,timeout=(12,20))
            assert r.status_code==206 and r.headers['Content-Range']==f"bytes 0-1048575/{meta['size_fields']['2']}"
            assert r.headers['ETag']==meta['headers']['Etag']
            body=r.content;r.close();assert len(body)==1048576
        record.update(status='ok',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),direct_seconds=time.monotonic()-download_start)
    except Exception as error:record.update(status='failed',error=repr(error))
    record['elapsed_seconds']=time.monotonic()-start
    (ROOT/'redirect_transport_probe.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='error'}))
