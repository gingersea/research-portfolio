"""Verify byte identity and cache behavior when reusing an issued content URL."""
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin,urlparse
import requests
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    meta=json.loads((ROOT.parent/'real_measurement_20261117/archive_metadata/range_probe.json').read_text())
    url=meta['strings'][1].replace('dl=0','dl=1')+'&probe=redirect_reuse_20261117'
    record={'global_config_changed':False,'steps':[]};begin=time.monotonic()
    try:
        with requests.Session() as s:
            for _ in range(8):
                host=urlparse(url).hostname;assert any(host==h or host.endswith('.'+h) for h in ['dropbox.com','dropboxusercontent.com'])
                r=s.get(url,headers={'Range':'bytes=0-1048575'},stream=True,allow_redirects=False,timeout=(12,25))
                if not r.is_redirect:break
                url=urljoin(url,r.headers['Location']);r.close()
            assert r.status_code==206;r.close()
            for slot in range(2):
                start=slot*(1<<20);end=start+(1<<20)-1;at=time.monotonic()
                r=s.get(url,headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'},stream=True,allow_redirects=False,timeout=(12,25))
                assert r.status_code==206
                assert r.headers['Content-Range']==f"bytes {start}-{end}/{meta['size_fields']['2']}"
                assert r.headers['ETag']==meta['headers']['Etag']
                body=r.content;r.close();assert len(body)==1<<20
                with Path('/data/AI_Research/shared/real_measurement_20261117/parts8/00000.part').open('rb') as f:
                    f.seek(start);assert f.read(1<<20)==body
                record['steps'].append({'slot':slot,'seconds':time.monotonic()-at,'sha256':hashlib.sha256(body).hexdigest(),'old_range_bytes_identical':True})
        record['status']='ok'
    except Exception as error:record.update(status='failed',error=repr(error))
    record['elapsed_seconds']=time.monotonic()-begin
    (ROOT/'reuse_redirect_probe.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
