"""Freeze final adapters and analysis while acquisition continues; no test scores seen."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha

if __name__=='__main__':
    files=[ROOT/name for name in ['run_native.py','run_descriptor.py','prepare_validation.py','analyze.py','pairing_audit.py',
           'PROTOCOL.md','implementation_assets.json','checkpoint.json']]
    files += [PROJECT/'benchmark/physcheck/relations'/name for name in ['descriptor.py','sparse.py','contracts.py']]
    files += [PROJECT/'research/tracking_validation_20261115/assets.json']
    assert not(ROOT/'method_freeze.json').exists()
    record={'utc':datetime.now(timezone.utc).isoformat(),'new_validation_predictions_or_scores_seen':False,
            'acquisition_still_running':True,'selection_rule_will_not_adapt_to_predictions':True,
            'sources':{str(p):sha(p) for p in files}}
    save(ROOT/'method_freeze.json',record)
    print(json.dumps({'frozen_files':len(files),'new_validation_scores_seen':False}))
