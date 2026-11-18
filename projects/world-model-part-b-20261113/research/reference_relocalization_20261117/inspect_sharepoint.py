"""Read public author sharing page directly, without account credentials."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin,urlparse

import requests

ROOT=Path(__file__).resolve().parent
URL='https://uob-my.sharepoint.com/:f:/g/personal/rg16964_bristol_ac_uk/IgAaTp6P4aRrSbUrYVa9EuC1ARZeGUGyCa5vPwdTvoflF0E?e=sWSIlO'

if __name__=='__main__':
    output=ROOT/'sharepoint_direct'
    output.mkdir(exist_ok=False)
    session=requests.Session()
    session.trust_env=False
    url=URL
    chain=[]
    for _ in range(8):
        host=urlparse(url).hostname
        assert host=='uob-my.sharepoint.com',host
        response=session.get(url,timeout=(15,60),allow_redirects=False)
        chain.append({'url':url,'status':response.status_code})
        if response.is_redirect:
            url=urljoin(url,response.headers['Location'])
            continue
        response.raise_for_status()
        path=output/'folder.html'
        path.write_bytes(response.content)
        (output/'receipt.json').write_text(json.dumps({'url':URL,'chain':chain,'bytes':len(response.content),
          'sha256':hashlib.sha256(response.content).hexdigest(),'anonymous':True,'proxy_defaults_unchanged':True},indent=2))
        print(json.dumps({'status':response.status_code,'bytes':len(response.content)}))
        break
