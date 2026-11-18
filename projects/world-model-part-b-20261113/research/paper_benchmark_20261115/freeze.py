"""Non-overwriting source/analysis/input receipt snapshot."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from fetch import sha

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage');args=parser.parse_args()
    out=ROOT/'source_snapshots'/args.stage;out.mkdir(parents=True,exist_ok=False)
    files={}
    for p in ROOT.iterdir():
        if p.suffix in ['.py','.json','.md']:
            shutil.copy2(p,out/p.name);files[p.name]=sha(p)
    deps={}
    for name in ['tracking_validation_20261115/assets.json','observation_gate_20261115/tapip_assets.json',
                 'observation_gate_20261115/tapip_source.json','identity_recovery_20261115/depth_assets.json']:
        deps[name]=sha(ROOT.parent/name)
    (out/'manifest.json').write_text(json.dumps({'stage':args.stage,'utc':datetime.now(timezone.utc).isoformat(),
        'files':files,'read_only_dependencies':deps},indent=2)+'\n')
    print(json.dumps({'stage':args.stage,'files':len(files)}))


if __name__=='__main__':main()
