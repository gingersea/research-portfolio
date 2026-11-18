"""Small pinned official reader resize dependency, isolated from prior environments."""
import hashlib
import json
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parent
URL='https://files.pythonhosted.org/packages/10/70/eeb41d81fea2b3165cfe7ae1c662f9dc6310b14b230a96260376c2891cc7/mediapy-1.2.7-py3-none-any.whl'
EXPECTED='fc81a9190d9724b02286efac4e7a67f384024e078fda0a3cac9fdf25e2f8878f'


def main():
    target=Path('/data/AI_Research/shared/paper_benchmark_20261115/wheels/mediapy-1.2.7-py3-none-any.whl')
    target.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(URL,timeout=30);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==EXPECTED
    if target.exists():assert target.read_bytes()==r.content
    else:target.write_bytes(r.content)
    (ROOT/'resize_dependency.json').write_text(json.dumps({'url':URL,'sha256':EXPECTED,'local':str(target),'version':'1.2.7'},indent=2)+'\n')


if __name__=='__main__':main()
