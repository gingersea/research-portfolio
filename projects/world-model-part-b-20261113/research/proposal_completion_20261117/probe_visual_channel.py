"""Known-color software fixture, not a physical benchmark or parameter tuning."""
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.judge.local_vlm import LocalVisualJudge
from physcheck.relations.contracts import save, sha


def main():
    assets = json.loads((ROOT / 'vlm_assets.json').read_text())
    judge = LocalVisualJudge(assets['directory'], 1)
    output = ROOT / 'visual_channel_probe'
    output.mkdir(exist_ok=False)
    records = []
    for expected, rgb in [('red', (255, 0, 0)), ('blue', (0, 0, 255)), ('green', (0, 255, 0))]:
        image = Image.new('RGB', (256, 256), rgb)
        image.save(output / (expected + '.png'))
        answer, runtime = judge.generate([image], 'Name the main color of this image. Reply with exactly one English color word.')
        records.append({'expected': expected, 'answer': answer, 'contains_expected': expected in answer.lower(),
             'runtime': runtime, 'image_sha256': sha(output / (expected + '.png'))})
    save(output / 'results.json', {'cases': records, 'all_color_correct': all(r['contains_expected'] for r in records),
         'scope': 'software image channel check; not physical judgment calibration', 'run_after_first_gravity_attempt': True})
    print(json.dumps({'all_color_correct': all(r['contains_expected'] for r in records)}))


if __name__ == '__main__':
    main()
