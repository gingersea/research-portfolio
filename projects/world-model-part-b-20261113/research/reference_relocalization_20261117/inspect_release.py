"""Pin primary implementation and inspect anonymous author weight access."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin,urlparse

import requests

ROOT=Path(__file__).resolve().parent
ALLOWED=['github.com','api.github.com','raw.githubusercontent.com','codeload.github.com',
         'uob-my.sharepoint.com','naver.com','naverlabs.com','huggingface.co','hf.co']


def get(url,limit=8_000_000):
    session=requests.Session()
    for _ in range(8):
        host=urlparse(url).hostname
        assert any(host==h or host.endswith('.'+h) for h in ALLOWED),f'Unapproved redirect: {host}'
        response=session.get(url,timeout=(15,60),allow_redirects=False,stream=True)
        if response.is_redirect:
            url=urljoin(url,response.headers['Location'])
            response.close()
            continue
        response.raise_for_status()
        value=bytearray()
        for block in response.iter_content(1<<16):
            value.extend(block)
            if len(value)>limit:
                raise ValueError('Metadata limit reached')
        return bytes(value),url,dict(response.headers)
    raise ValueError('Redirect limit')


if __name__=='__main__':
    output=ROOT/'sources'
    output.mkdir(exist_ok=False)
    raw,_,_=get('https://api.github.com/repos/rhodriguerrier/PointSt3R/commits/main')
    (output/'revision.json').write_bytes(raw)
    revision=json.loads(raw)['sha']
    files=['README.md','.gitmodules','pointst3r_ego_points_eval.py','mast3r/model.py','mast3r/fast_nn.py']
    tasks={name:f'https://raw.githubusercontent.com/rhodriguerrier/PointSt3R/{revision}/{name}' for name in files}
    tasks['sharepoint.html']='https://uob-my.sharepoint.com/:f:/g/personal/rg16964_bristol_ac_uk/IgAaTp6P4aRrSbUrYVa9EuC1ARZeGUGyCa5vPwdTvoflF0E?e=sWSIlO'
    def one(item):
        name,url=item
        try:
            content,final,headers=get(url)
            target=output/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
            return {'file':name,'url':url,'final_url':final,'status':'ok','bytes':len(content),
                    'sha256':hashlib.sha256(content).hexdigest()}
        except Exception as error:
            return {'file':name,'url':url,'status':'failed','error':repr(error)}
    with ThreadPoolExecutor(6) as pool:
        rows=list(pool.map(one,tasks.items()))
    (output/'receipt.json').write_text(json.dumps({'revision':revision,'files':rows},indent=2))
    print(json.dumps(rows))
