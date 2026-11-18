"""Copy source/assets/caches and actually rescore from the copied workspace."""
from datetime import datetime,timezone
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
    else:shutil.copy2(source,target)

def main():
    before=ROOT/'source_snapshots/README.before.md'
    diff=''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),
        (PROJECT/'README.md').read_text().splitlines(keepends=True),fromfile='README.before.md',tofile='README.md'))
    with (ROOT/'source_snapshots/README.diff').open('x') as f:f.write(diff)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bundle=Path('/data/AI_Research/shared/handoff')/('world_model_reference_relocalization_20261117_v1_'+stamp)
    bundle.mkdir(exist_ok=False)
    work=bundle/'project';target=work/'research'/ROOT.name
    copy(PROJECT/'benchmark/physcheck',work/'benchmark/physcheck')
    copy(ROOT,target)
    copy(PROJECT/'README.md',work/'README.md')
    for name in ['test_descriptor_relocalization.py','test_relocalization_analysis.py']:
        copy(PROJECT/'tests'/name,work/'tests'/name)
    prior=PROJECT/'research/real_measurement_20261117';copied_prior=work/'research'/prior.name
    copy(prior/'manifest.json',copied_prior/'manifest.json')
    copy(prior/'predictions_memory',copied_prior/'predictions_memory')
    old=json.loads((prior/'manifest.json').read_text())
    for case in old['cases']:
        for key in ['truth','annotation','source_images']:
            record=case['files'][key]
            copy(Path(record['path']),copied_prior/'data'/case['id']/Path(record['path']).name)
    assets=json.loads((ROOT/'implementation_assets.json').read_text())
    weights=json.loads((ROOT/'checkpoint.json').read_text())
    co=json.loads((PROJECT/'research/tracking_validation_20261115/assets.json').read_text())
    selection=json.loads((ROOT/'selection.json').read_text())
    for source,name in [(Path(assets['source_directory']),'PointSt3R_source'),(Path(assets['archive']),'PointSt3R_source.zip'),
          (Path(weights['checkpoint']),'MASt3R.pth'),(Path(co['source_directory']),'CoTracker3_source'),
          (Path(co['checkpoint']),'CoTracker3.pth'),(Path(selection['archive']),'ego_points_prefix.tar')]:
        copy(source,bundle/'assets'/name)
    save(bundle/'ASSETS.json',{'original_pointst3r_source':assets,'original_mast3r_weights':weights,
        'original_cotracker':co,'raw_prefix':selection,'fresh_gpu_replay_claim':False,
        'old_development_rgb_inputs_copied':False,'new_validation_inputs_copied':True})
    commands=[];comparisons=[]
    splits=[('development',copied_prior/'manifest.json',copied_prior/'predictions_memory',
             target/'predictions_development',copied_prior/'data','analysis_development','pairing_development')]
    validation=json.loads((target/'manifest_validation.json').read_text())
    if validation['cases']:
        splits.append(('new_participant_validation',target/'manifest_validation.json',target/'predictions_native_validation',
                       target/'predictions_descriptor_validation',target/'data_validation','analysis_validation','pairing_validation'))
    for split,manifest,native,descriptor,data,analysis,pairing in splits:
        for script,out,original in [('analyze.py',analysis+'_replay',analysis),('pairing_audit.py',pairing+'_replay',pairing)]:
            cmd=[sys.executable,str(target/script),'--manifest',str(manifest),'--native',str(native),
                 '--descriptor',str(descriptor),'--relocated-data',str(data),'--output',str(target/out)]
            if script=='analyze.py':cmd+=['--split',split]
            result=subprocess.run(cmd,cwd=work,text=True,capture_output=True)
            commands.append({'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            assert result.returncode==0,commands[-1]
            files=['summary.json','per_video.json']+(['per_video.csv'] if script=='analyze.py' else ['events.json'])
            for name in files:
                same=(target/out/name).read_bytes()==(target/original/name).read_bytes()
                assert same,(out,name)
                comparisons.append({'replay':out+'/'+name,'original':original+'/'+name,'bytes_identical':same,'sha256':sha(target/out/name)})
    save(target/'handoff_receipt.json',{'bundle':str(bundle),'copied_scoring_and_pairing_byte_identical':True,
        'comparisons':comparisons,'bundle_integrity_receipt':str(bundle/'RECEIPT.json'),
        'fresh_gpu_inference_replay':False,'public_upload':False})
    hashes={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
    save(bundle/'HASHES.json',hashes)
    for relative,digest in hashes.items():assert sha(bundle/relative)==digest
    receipt={'bundle':str(bundle),'files_hashed':len(hashes),'bytes':sum((bundle/p).stat().st_size for p in hashes),
        'all_file_hashes_verified':True,'commands':commands,'copied_source_cache_replay':comparisons,
        'fresh_gpu_inference_replay':False,'original_truth_access_required_for_replay':False,
        'excluded_old_rgb_input_bytes':True,'public_upload':False}
    save(bundle/'RECEIPT.json',receipt)
    save(ROOT/'handoff_receipt.json',{'bundle':str(bundle),'receipt_sha256':sha(bundle/'RECEIPT.json'),
        'files_hashed':len(hashes),'bytes':receipt['bytes'],'copied_scoring_and_pairing_byte_identical':True,
        'all_file_hashes_verified':True,'fresh_gpu_replay':False,'public_upload':False})
    print(json.dumps({'bundle':str(bundle),'files':len(hashes),'bytes':receipt['bytes'],'replay_identical':True}))

if __name__=='__main__':main()
