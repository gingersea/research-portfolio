"""Same planned URLs, alternative direct transport; preserve all first attempts."""
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import cv2
import requests

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha


def main():
    plan = json.loads((ROOT / 'human_probe_plan.json').read_text())['rows']
    output = ROOT / 'human_videos_direct'
    output.mkdir(exist_ok=False)
    for row in plan:
        prior = ROOT / 'human_videos' / (row['opaque_id'] + '.json')
        if prior.exists() and json.loads(prior.read_text())['status'] == 'ok':
            continue
        receipt = {'opaque_id': row['opaque_id'], 'source_url': row['url'], 'transport': 'direct, same original URL',
                   'previous_attempt': str(prior), 'replacement_sample': False}
        try:
            assert not any(x in urlparse(row['url']).netloc for x in ['google', 'gingersea'])
            session = requests.Session()
            session.trust_env = False
            response = session.get(row['url'], stream=True, timeout=(15, 60), allow_redirects=False)
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError('Unexpected redirect/range response')
            path = output / (row['opaque_id'] + '.mp4')
            with path.open('xb') as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
            cap = cv2.VideoCapture(str(path))
            count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            cap.release()
            assert count > 1 and fps > 0
            receipt.update(status='ok', file=str(path), sha256=sha(path), frames=count, fps=fps, bytes=path.stat().st_size)
        except Exception as error:
            receipt.update(status='retrieval_failed', error=repr(error), planned_slot_retained=True)
        save(output / (row['opaque_id'] + '.json'), receipt)
        print(json.dumps({'video': row['opaque_id'], 'status': receipt['status']}), flush=True)


if __name__ == '__main__':
    main()
