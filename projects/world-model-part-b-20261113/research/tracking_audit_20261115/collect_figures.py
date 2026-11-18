"""Archive selected primary figures for direct method inspection, unmodified."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

from diagnose import sha

ROOT = Path(__file__).resolve().parent
FIGURES = [('spatialtracker', 'S1.F2'), ('cotracker3', 'S3.F2'),
           ('megasam_v2', 'S3.F4'), ('tapip3d_v3', 'S3.F2'), ('tapip3d_v3', 'S3.F3')]


def fetch(item):
    name, identifier = item
    soup = BeautifulSoup((ROOT / 'papers' / name / 'paper.html').read_text(), 'html.parser')
    element = soup.find(id=identifier)
    source = element.find('img')['src']
    url = urljoin('https://arxiv.org/html/', source)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination = ROOT / 'papers' / name / f'{identifier}.png'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(response.content)
    return {'paper': name, 'figure': identifier, 'url': url,
            'sha256': sha(destination), 'caption': element.get_text(' ', strip=True)}


if __name__ == '__main__':
    rows = list(ThreadPoolExecutor(max_workers=5).map(fetch, FIGURES))
    (ROOT / 'figure_sources.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'caption'} for row in rows], indent=2))
