"""Archive public primary HTML papers and preserve their mathematical text."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parent
PAPERS = {
    'spatialtracker': '2404.04319v1',
    'cotracker3': '2410.11831v1',
    'megasam': '2412.04463v1',
    'tapip3d': '2504.14717v1',
}


def fetch(item):
    name, identifier = item
    target = ROOT / 'papers' / name
    target.mkdir(parents=True, exist_ok=False)
    url = f'https://arxiv.org/html/{identifier}'
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    content = response.content
    (target / 'paper.html').write_bytes(content)
    soup = BeautifulSoup(content, 'html.parser')
    article = soup.find('article') or soup
    for element in article.find_all('math'):
        element.replace_with(' $' + element.get('alttext', element.get_text(' ', strip=True)) + '$ ')
    for element in article.find_all(['script', 'style']):
        element.decompose()
    text = article.get_text(' ', strip=True)
    (target / 'paper.txt').write_text(text + '\n')
    # Separate semantic blocks permit complete reading without dumping references.
    blocks = []
    for element in article.find_all(['section']):
        if element.find_parent('section') is None:
            blocks.append({'id': element.get('id'), 'text': element.get_text(' ', strip=True)})
    (target / 'sections.json').write_text(json.dumps(blocks, indent=2, ensure_ascii=False) + '\n')
    return {'name': name, 'arxiv_version': identifier, 'url': url,
            'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'html_sha256': hashlib.sha256(content).hexdigest(),
            'text_sha256': hashlib.sha256((target / 'paper.txt').read_bytes()).hexdigest(),
            'characters': len(text), 'top_level_blocks': len(blocks)}


def main():
    rows = list(ThreadPoolExecutor(max_workers=4).map(fetch, PAPERS.items()))
    (ROOT / 'paper_sources.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
