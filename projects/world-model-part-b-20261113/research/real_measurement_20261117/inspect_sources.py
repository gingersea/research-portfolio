"""Small primary-source acquisition; metadata inspection before sample freezing."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent
ALLOWED = ['github.com', 'api.github.com', 'raw.githubusercontent.com', 'dropbox.com', 'dropboxusercontent.com', 'arxiv.org']


def get(url, max_bytes=5_000_000):
    session = requests.Session()
    for _ in range(8):
        host = urlparse(url).hostname
        assert any(host == h or host.endswith('.'+h) for h in ALLOWED), host
        response = session.get(url, timeout=(20,60), allow_redirects=False, stream=True)
        if response.is_redirect:
            from urllib.parse import urljoin
            url = urljoin(url, response.headers['Location'])
            response.close()
            continue
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(1<<16):
            content.extend(block)
            if len(content)>max_bytes:
                raise ValueError('Metadata size limit reached')
        return bytes(content), url, dict(response.headers)
    raise ValueError('Redirect limit')


def main():
    folder = ROOT/'sources'
    folder.mkdir(exist_ok=False)
    api = 'https://api.github.com/repos/AhmadDarKhalil/EgoPoints/commits/main'
    raw, _, _ = get(api)
    (folder/'github_revision.json').write_bytes(raw)
    revision = json.loads(raw)['sha']
    urls = {name: f'https://raw.githubusercontent.com/AhmadDarKhalil/EgoPoints/{revision}/{name}' for name in
            ['README.md', 'egopoints_dataloader.py', 'egopoints_eval_cotracker.py', 'visualise_egopoints_sequence.py']}
    urls['dropbox.html'] = 'https://www.dropbox.com/scl/fo/tfvctluqu3cr17jr6q0td/AA6h6GlV-x6QeuupmeLejzA?dl=0&rlkey=r0q12vbi6wour6qsteklivb6p&st=1e4b4dnn'
    def one(item):
        name, url = item
        try:
            content, final, headers = get(url)
            (folder/name).write_bytes(content)
            return {'file':name, 'url':url, 'final_url':final, 'bytes':len(content),
                    'sha256':hashlib.sha256(content).hexdigest(), 'status':'ok'}
        except Exception as error:
            return {'file':name, 'url':url, 'status':'failed', 'error':repr(error)}
    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(one, urls.items()))
    (folder/'receipt.json').write_text(json.dumps({'github_revision':revision,'files':results},indent=2))
    print(json.dumps(results))


if __name__=='__main__':
    main()
