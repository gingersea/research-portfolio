"""Immutable source snapshots, including external read-only baseline dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    destination = ROOT/'source_snapshots'/args.stage
    destination.mkdir(parents=True, exist_ok=False)
    sources = sorted([*ROOT.glob('*.py'), ROOT/'PROTOCOL.md'])
    hashes = {}
    for source in sources:
        shutil.copy2(source, destination/source.name)
        hashes[source.name] = sha(source)
    baseline = {str(path.relative_to(ROOT.parent.parent)): sha(path) for path in [
        ROOT.parent/'principia_i2v_20261115/evaluate.py', ROOT.parent/'principia_i2v_20261115/pilot_gate.json',
        ROOT.parent/'tracking_audit_20261115/candidate.py', ROOT.parent/'tracking_audit_20261115/identity_challenge_v1/result.json']}
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage, 'sha256': hashes,
                'read_only_baseline_sha256': baseline}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
