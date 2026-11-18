"""Record the resource-only amendment before corrected inference or GT scoring."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert not (ROOT/'analysis').exists()
    record={'utc':datetime.now(timezone.utc).isoformat(),'runner_sha256':sha(ROOT/'run_models_memory.py'),
         'manifest_sha256':sha(ROOT/'manifest.json'),'scorer_sha256':sha(ROOT/'analyze.py'),
         'amendment_sha256':sha(ROOT/'MEMORY_EXECUTION.md'),'human_reference_scores_unseen':True,
         'first_attempt_count':len(list((ROOT/'predictions').glob('*.json')))}
    with (ROOT/'memory_execution_freeze.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(record))
