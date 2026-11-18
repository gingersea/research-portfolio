"""Non-overwriting snapshots of executed pilot sources and protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

STUDY = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    if not args.stage.replace('_', '').isalnum():
        raise ValueError('Invalid stage name')
    target = STUDY / 'source_snapshots' / args.stage
    target.mkdir(parents=True, exist_ok=False)
    records = {}
    for source in sorted([*STUDY.glob('*.py'), STUDY / 'PROTOCOL.md']):
        shutil.copy2(source, target / source.name)
        records[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (target / 'manifest.json').write_text(json.dumps({'stage': args.stage, 'utc': datetime.now(timezone.utc).isoformat(), 'sha256': records}, indent=2) + '\n')
    print(json.dumps({'snapshot': str(target), 'files': len(records)}))


if __name__ == '__main__':
    main()
