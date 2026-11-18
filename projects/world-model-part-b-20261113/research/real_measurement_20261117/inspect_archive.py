"""Decode public folder metadata and probe archive range support, no video scores."""
import base64
import json
from pathlib import Path
import re
from urllib.parse import urlparse, urljoin

import requests

from inspect_sources import ROOT, ALLOWED, get


def varint(data, pos):
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError('Invalid varint')


def fields(data):
    pos = 0
    result = []
    while pos<len(data):
        tag,pos = varint(data,pos)
        number,wire = tag>>3,tag&7
        if not number:
            raise ValueError('Invalid field')
        if wire==0:
            value,pos = varint(data,pos)
        elif wire==2:
            size,pos = varint(data,pos)
            value = data[pos:pos+size]
            assert len(value)==size
            pos += size
        else:
            raise ValueError('Unsupported field')
        result.append((number,wire,value))
    return result


def main():
    folder = ROOT/'archive_metadata'
    folder.mkdir(exist_ok=False)
    html = (ROOT/'sources/dropbox.html').read_text()
    payloads = re.findall(r'registerStreamedPrefetch\("[^"]+", "([^"]+)"',html)
    candidates = []
    def visit(data, depth=0):
        if depth>12:
            return
        try:
            rows = fields(data)
        except Exception:
            return
        names = [v.decode('utf-8',errors='ignore') for _,w,v in rows if w==2]
        if any(s=='ego_points.tar' for s in names):
            candidates.append({'size_fields': {str(n):v for n,w,v in rows if w==0},
                  'strings': [s for s in names if s.startswith('http') or s=='ego_points.tar']})
        for n,w,value in rows:
            if w==2:
                visit(value,depth+1)
    for i,encoded in enumerate(payloads):
        raw = base64.b64decode(encoded)
        (folder/f'prefetch_{i}.bin').write_bytes(raw)
        visit(raw)
    (folder/'candidates.json').write_text(json.dumps(candidates,indent=2))
    record = candidates[0]
    url = next(s for s in record['strings'] if s.endswith('dl=0')).replace('dl=0','dl=1')
    session = requests.Session()
    headers = {'Range':'bytes=0-4095', 'Accept-Encoding':'identity'}
    for _ in range(8):
        host = urlparse(url).hostname
        assert any(host==h or host.endswith('.'+h) for h in ALLOWED)
        response = session.get(url, headers=headers, stream=True, timeout=(20,90), allow_redirects=False)
        if response.is_redirect:
            url=urljoin(url,response.headers['Location'])
            response.close()
            continue
        record.update(url=url,status=response.status_code,headers=dict(response.headers))
        # Do not accidentally download a large archive during inspection.
        block = next(response.iter_content(4096))
        (folder/'first4096.bin').write_bytes(block)
        response.close()
        break
    (folder/'range_probe.json').write_text(json.dumps(record,indent=2))
    for key in ['content-length','content-range','content-type']:
        print(key,record['headers'].get(key))
    print(json.dumps({'size_fields':record['size_fields'],'status':record['status'],'strings':record['strings']}))


if __name__=='__main__':
    main()
