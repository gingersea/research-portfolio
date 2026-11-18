"""Official API fallback after transient raw-host TLS failures."""
import base64
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT=Path(__file__).resolve().parent


def request(url):
    errors=[]
    for _ in range(3):
        try:
            response=requests.get(url,timeout=(15,40))
            response.raise_for_status()
            return response.json()
        except Exception as e:
            errors.append(repr(e))
            time.sleep(1)
    raise RuntimeError(errors)


if __name__=='__main__':
    folder=ROOT/'source_api'
    folder.mkdir(exist_ok=False)
    rev=json.loads((ROOT/'sources/revision.json').read_text())['sha']
    rows=[]
    for name in ['README.md','.gitmodules','pointst3r_ego_points_eval.py','mast3r/model.py','mast3r/fast_nn.py']:
        try:
            item=request(f'https://api.github.com/repos/rhodriguerrier/PointSt3R/contents/{name}?ref={rev}')
            content=base64.b64decode(item['content'])
            assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==item['sha']
            path=folder/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
            rows.append({'file':name,'status':'ok','sha256':hashlib.sha256(content).hexdigest(),'git_blob_sha':item['sha']})
        except Exception as e:
            rows.append({'file':name,'status':'failed','error':repr(e)})
    model=request('https://huggingface.co/api/models/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric?blobs=true')
    (folder/'mast3r_model.json').write_text(json.dumps(model,indent=2))
    (folder/'receipt.json').write_text(json.dumps({'revision':rev,'files':rows},indent=2))
    print(json.dumps({'files':rows,'hf_revision':model['sha'],'hf_files':model['siblings']}))
