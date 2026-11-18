"""Direct public wheel transfer with PyPI hash verification for the isolated env."""
import json
from pathlib import Path

import requests

from fetch_tapip import ASSETS,ROOT,sha


if __name__=='__main__':
    target=ASSETS/'wheels';target.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    rows=[]
    for package,version in [('hydra-core','1.3.2'),('python-box','7.3.2')]:
        response=session.get(f'https://pypi.org/pypi/{package}/{version}/json',timeout=15);response.raise_for_status()
        candidates=response.json()['urls']
        item=next(item for item in candidates if item['filename'].endswith('py3-none-any.whl') or
                  'cp312-cp312-manylinux_2_17_x86_64' in item['filename'])
        path=target/item['filename']
        if not path.exists():
            response=session.get(item['url'],timeout=30);response.raise_for_status();path.write_bytes(response.content)
        assert sha(path)==item['digests']['sha256']
        rows.append({'package':package,'version':version,'filename':str(path),'sha256':sha(path),'url':item['url']})
    (ROOT/'environment_wheels.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
