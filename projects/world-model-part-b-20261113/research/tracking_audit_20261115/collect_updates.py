"""Keep initial versions, and archive the verified latest revisions separately."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from collect_papers import fetch

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    rows = list(ThreadPoolExecutor(max_workers=2).map(fetch, [
        ('megasam_v2', '2412.04463v2'), ('tapip3d_v3', '2504.14717v3')]))
    (ROOT / 'paper_sources_updates.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
