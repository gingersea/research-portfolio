"""Create a non-overwriting source snapshot at each declared study stage."""
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
        raise ValueError('Stage must be an identifier')
    destination = STUDY / 'source_snapshots' / args.stage
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for source in sorted([*STUDY.glob('*.py'), STUDY / 'PROTOCOL.md']):
        target = destination / source.name
        shutil.copy2(source, target)
        records[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / 'manifest.json').write_text(json.dumps({'stage': args.stage, 'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': records}, indent=2) + '\n')
    print(json.dumps({'snapshot': str(destination), 'files': len(records)}))


if __name__ == '__main__':
    main()
