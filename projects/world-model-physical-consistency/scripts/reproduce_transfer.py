"""Adapt archived absolute video paths in a temporary workspace for replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        parser.error('--output must be a new directory')
    records = json.loads((root/'data/video_manifest.json').read_text())
    video_paths = {(r['model'],r['prompt_id'],r['seed']):str(root/r['path']) for r in records}
    with tempfile.TemporaryDirectory(prefix='physcheck-transfer-') as temporary:
        source = Path(temporary)
        raw = source/'benchmark/results/raw'
        raw.mkdir(parents=True)
        shutil.copy2(root/'benchmark/training_pilot.py',source/'benchmark/training_pilot.py')
        shutil.copytree(root/'benchmark/results/training_pilot',source/'benchmark/results/training_pilot')
        rows = [json.loads(line) for line in (root/'benchmark/results/raw/part_a.jsonl').read_text().splitlines()]
        for row in rows:
            row['video_path'] = video_paths[(row['model'],row['prompt_id'],row['seed'])]
        # This is a derived path mapping, never a rewrite of the frozen JSONL.
        (raw/'part_a.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        subprocess.run([sys.executable,str(root/'research/transfer_20261109T143307Z/screen_transfer.py'),
                        '--source',str(source),'--output',str(args.output.resolve())],check=True)
    print('Portable transfer replay completed; archived JSONL was not modified.')


if __name__ == '__main__':
    main()
