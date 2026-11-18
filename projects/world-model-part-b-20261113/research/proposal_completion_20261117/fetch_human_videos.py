"""Prespecified stratified PC probe; retrieval failures retain planned slots."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse, urljoin

import cv2
import requests

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha


def forbidden(url):
    host = (urlparse(url).hostname or '').lower()
    return any(s in host for s in ['google', 'gingersea']) or urlparse(url).scheme != 'https'


def retrieve(record):
    url = record['url']
    target = ROOT / 'human_videos' / (record['opaque_id'] + '.mp4')
    receipt = {'opaque_id': record['opaque_id'], 'source_url': url}
    try:
        for _ in range(8):
            if forbidden(url):
                raise ValueError('Forbidden external host')
            response = requests.get(url, allow_redirects=False, stream=True, timeout=(20, 90))
            if response.status_code in [301, 302, 303, 307, 308]:
                url = urljoin(url, response.headers['Location'])
                response.close()
                continue
            response.raise_for_status()
            with target.open('xb') as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
            response.close()
            break
        else:
            raise ValueError('Too many redirects')
        cap = cv2.VideoCapture(str(target))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames += 1
        cap.release()
        if frames < 2 or fps <= 0:
            raise ValueError('No decodable temporal sequence')
        receipt.update(status='ok', file=str(target), sha256=sha(target), bytes=target.stat().st_size,
                       frames=frames, fps=fps)
    except Exception as error:
        receipt.update(status='retrieval_failed', error=repr(error), planned_slot_retained=True)
    save(target.with_suffix('.json'), receipt)
    print(json.dumps({'video': record['opaque_id'], 'status': receipt['status']}), flush=True)
    return receipt


def main():
    source = Path('/data/AI_Research/shared/physics_methods_20261114/data/videophy2_test/videophy2_test.csv')
    assert sha(source) == '85a6690b9508b7e69c592f3cbcbc4113efd3a573eb5ec69d6ae030a8ffb8a4e7'
    rows = list(csv.DictReader(source.open()))
    plan = []
    for consistent in [False, True]:
        candidates = [r for r in rows if (int(r['pc']) >= 4 if consistent else int(r['pc']) <= 2) and not forbidden(r['video_url'])]
        candidates.sort(key=lambda r: hashlib.sha256(('proposal20261117:' + r['video_url']).encode()).hexdigest())
        for i, row in enumerate(candidates[:12]):
            plan.append({'opaque_id': 'h_' + hashlib.sha256(row['video_url'].encode()).hexdigest()[:16],
                 'url': row['video_url'], 'split': 'calibration' if i < 4 else 'evaluation',
                 'human_pc': int(row['pc']), 'consistent_reference': consistent,
                 'caption': row['caption'], 'model': row['model_name'], 'action': row['action']})
    assert len(plan) == 24 and len({r['opaque_id'] for r in plan}) == 24
    save(ROOT / 'human_probe_plan.json', {'rows': plan, 'source_sha256': sha(source),
         'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'labels_not_sent_to_judge': True})
    (ROOT / 'human_videos').mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(retrieve, plan))
    save(ROOT / 'human_probe_retrieval.json', {'planned': 24, 'obtained': sum(r['status'] == 'ok' for r in results), 'cases': results})


if __name__ == '__main__':
    main()
