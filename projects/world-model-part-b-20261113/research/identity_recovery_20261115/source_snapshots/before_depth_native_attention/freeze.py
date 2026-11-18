"""Versioned source and dependency hashes; never replace an earlier stage."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage');args=parser.parse_args()
    target=ROOT/'source_snapshots'/args.stage;target.mkdir(parents=True,exist_ok=False)
    current={}
    for path in ROOT.iterdir():
        if path.suffix in ['.py','.json','.md']:
            shutil.copy2(path,target/path.name);current[path.name]=sha(path)
    dependencies={}
    for relative in ['observation_gate_20261115/gate.py','observation_gate_20261115/new_controls.py',
                     'observation_gate_20261115/run_model.py','tracking_validation_20261115/render_controls.py',
                     'tracking_validation_20261115/metrics.py','principia_i2v_20261115/input_manifest.json']:
        dependencies[relative]=sha(ROOT.parent/relative)
    (target/'manifest.json').write_text(json.dumps({'stage':args.stage,'utc':datetime.now(timezone.utc).isoformat(),
        'files':current,'read_only_dependencies':dependencies},indent=2)+'\n')
    print(json.dumps({'stage':args.stage,'files':len(current),'read_only_dependencies':len(dependencies)}))


if __name__=='__main__':main()
