"""Join the real isolated judgments and invoke the calibration-gated CLI.

Failed calibration is a result: retain status/diagnostics, never write a valid
score by silently skipping controls. This script cannot invent judge outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'benchmark'))
from physcheck.judge.production import validate_records


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=STUDY/'run')
    parser.add_argument('--phase',choices=['calibration','production'],required=True)
    parser.add_argument('--results-dir',type=Path,help='New checkpoint root; preserves the original results')
    args=parser.parse_args()
    run=args.run.resolve()
    records=[]
    for model in ['wan2.1_1.3b','cogvideox_5b']:
        records.extend(json.loads((run/'private'/f'{model}.json').read_text()))
    records=[r for r in records if args.phase=='production' or r['is_calibration']]
    assert len(records)==(56 if args.phase=='production' else 8)
    assert all(r['status']=='ok' for r in records)
    labels_path=run/'private/control_eligibility.json'
    eligibility=json.loads(labels_path.read_text()) if labels_path.exists() else {}
    receipt_path=run/'judge_session_receipts.json'
    receipts=json.loads(receipt_path.read_text())['receipts']
    manifest={}
    outputs=[]
    for record in records:
        vid=record['video_id']
        assert hashlib.sha256(Path(record['video_path']).read_bytes()).hexdigest()==record['video_sha256']
        info={k:record[k] for k in ['model','prompt_id','seed','kind','is_calibration',
                                    'expected_violation_kind','sampled_frame_indices']}
        if info['is_calibration']:
            info['calibration_observable']=eligibility.get(vid,{}).get('eligible',False)
            info['control_eligibility']=eligibility.get(vid,{'reason':'no independent image-based eligibility record'})
        manifest[vid]=info
        judge_path=run/'judgments'/f'{vid}.json'
        raw=judge_path.read_text()
        receipt=receipts[vid]
        assert receipt['judgment_sha256']==hashlib.sha256(judge_path.read_bytes()).hexdigest()
        assert receipt['fork_turns']=='none' and receipt['image_output_events']>0
        assert receipt['grid_sha256']==hashlib.sha256(Path(record['grid_path']).read_bytes()).hexdigest()
        assert receipt['archived_grid_was_actually_viewed'] is True
        outputs.append({'video_id':vid,'raw_output':raw,'judge_model':receipt['model'],
            'judge_version':'gpt-6-astra harness alias; dated model snapshot undisclosed',
            'judge_session':receipt['session_id'],'judge_agent_path':receipt['agent_path'],
            'judge_role':receipt['agent_role'],
            'judge_output_sha256':hashlib.sha256(judge_path.read_bytes()).hexdigest(),
            'judge_grid_sha256':hashlib.sha256(Path(record['grid_path']).read_bytes()).hexdigest()})
    validation=validate_records(outputs,manifest,calibration_only=args.phase=='calibration')
    if args.phase=='calibration':
        dump(run/'calibration_gate.json',validation)
        print(json.dumps(validation['calibration'],indent=2))
        return
    results=(args.results_dir or run/'results').resolve()
    receipt_dir=results if args.results_dir else run
    raw_dir=results/'raw';raw_dir.mkdir(parents=True,exist_ok=True)
    dump(raw_dir/'part_b_manifest.json',manifest)
    (raw_dir/'judge_outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
    env={**os.environ,'PYTHONPATH':str(ROOT/'benchmark'),'PHYSCHECK_RESULTS_DIR':str(results),
         'PHYSCHECK_ARTIFACTS_DIR':str(run/'private/artifacts'),
         'PHYSCHECK_MODELS_DIR':str(run/'private/unused_models'),'PYTHONDONTWRITEBYTECODE':'1'}
    command=[sys.executable,'-m','physcheck','judge','--input-file',str(raw_dir/'judge_outputs.jsonl'),
             '--manifest',str(raw_dir/'part_b_manifest.json'),'--output-file',str(raw_dir/'judge_validated.jsonl')]
    result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    dump(receipt_dir/'judge_cli_result.json',{'command':command,'returncode':result.returncode,
                                     'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode==(0 if validation['calibration']['valid'] else 2),result.stderr
    result2=subprocess.run([sys.executable,'-m','physcheck','report','--part','b'],
                           cwd=ROOT,env=env,text=True,capture_output=True)
    dump(receipt_dir/'report_cli_result.json',{'returncode':result2.returncode,'stdout':result2.stdout,'stderr':result2.stderr})
    assert result2.returncode==0,result2.stderr
    print(result.stdout)
    print(result2.stdout)


if __name__=='__main__':main()
