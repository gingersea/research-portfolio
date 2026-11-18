"""Local source/results handoff with actual copied-source sparse evaluation."""
from datetime import datetime,timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=Path('/data/AI_Research/shared/handoff')/('world_model_real_measurement_20261117_v1_'+stamp)
    target.mkdir()
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache')
    copied=target/'research'/ROOT.name
    shutil.copytree(ROOT,copied,ignore=ignore)
    shutil.copytree(PROJECT/'benchmark/physcheck',target/'benchmark/physcheck',ignore=ignore)
    for path in [PROJECT/'tests/test_sparse_measurement.py',PROJECT/'research/tracking_validation_20261115/assets.json',
                 PROJECT/'README.md']:
        destination=target/path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    # Retain original frozen manifest and create a separate relocated evaluator copy.
    frozen=copied/'manifest.json'
    frozen.rename(copied/'manifest.original.json')
    manifest=json.loads((copied/'manifest.original.json').read_text())
    for case in manifest['cases']:
        for entry in case['files'].values():
            old=Path(entry['path'])
            destination=target/old.relative_to(PROJECT)
            assert destination.exists() and sha(destination)==entry['sha256']
            entry['path']=str(destination)
    save(copied/'manifest.json',manifest)
    env=dict(os.environ,PYTHONPATH=str(target/'benchmark'))
    replay=copied/'copy_replay'
    command=[sys.executable,str(copied/'analyze.py'),'--predictions',str(copied/'predictions_memory'),'--output',str(replay)]
    result=subprocess.run(command,cwd=target,env=env,capture_output=True,text=True)
    save(target/'REPLAY_COMMAND.json',{'command':command,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode==0,result.stderr
    for name in ['summary.json','per_video.json']:
        assert json.loads((replay/name).read_text())==json.loads((ROOT/'analysis'/name).read_text()),name
    first_replay=copied/'copy_replay_first'
    result=subprocess.run([sys.executable,str(copied/'analyze.py'),'--predictions',str(copied/'predictions'),
          '--output',str(first_replay)],cwd=target,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for name in ['summary.json','per_video.json']:
        assert json.loads((first_replay/name).read_text())==json.loads((ROOT/'analysis_first_attempt'/name).read_text()),name
    regenerated=copied/'copy_report'
    result=subprocess.run([sys.executable,str(copied/'report.py'),'--output',str(regenerated)],cwd=target,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for name in ['REPORT.zh.md','per_video.csv']:
        assert (regenerated/name).read_bytes()==(ROOT/'report'/name).read_bytes(),name
    save(target/'COPY_REPLAY.json',{'all_case_metrics_and_summary_identical':True,'first_attempt_including_failures_identical':True,'report_and_csv_byte_identical':True,
         'clips':len(manifest['cases']),'fresh_neural_rerun':False,'scope':'Copied source/inputs/predictions, existing local environment; original raw JPEG archive and weights external'})
    additions=[PROJECT/'benchmark/physcheck/relations/sparse.py',PROJECT/'tests/test_sparse_measurement.py']
    diff=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(PROJECT)))) for p in additions)
    (target/'SOURCE_ADDITIONS.diff').write_text(diff)
    before_readme=ROOT/'source_snapshots/README.before.md'
    (target/'README_CHANGE.diff').write_text(''.join(difflib.unified_diff(before_readme.read_text().splitlines(True),
        (PROJECT/'README.md').read_text().splitlines(True),fromfile='README.before.md',tofile='README.md')))
    before=(ROOT/'run_models.py').read_text().splitlines(True)
    after=(ROOT/'run_models_memory.py').read_text().splitlines(True)
    (target/'RESOURCE_CHANGE.diff').write_text(''.join(difflib.unified_diff(before,after,fromfile='run_models.py',tofile='run_models_memory.py')))
    with zipfile.ZipFile(target/'source_bundle.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for base in [target/'benchmark/physcheck',target/'tests',copied]:
            for p in base.rglob('*.py'):
                if '__pycache__' not in p.parts:
                    archive.write(p,p.relative_to(target))
    files={str(p.relative_to(target)):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    save(target/'MANIFEST.json',{'files':files,'public_upload':False,'raw_archive_and_model_weights_external':True})
    assert all(sha(target/name)==record['sha256'] for name,record in files.items())
    save(ROOT/'handoff_receipt.json',{'path':str(target),'files':len(files),'bytes':sum(r['bytes'] for r in files.values()),
         'all_hashes_verified':True,'copied_evaluation_identical':True,'copied_report_byte_identical':True,
         'manifest_sha256':sha(target/'MANIFEST.json'),'new_public_upload':False})
    print((ROOT/'handoff_receipt.json').read_text())
