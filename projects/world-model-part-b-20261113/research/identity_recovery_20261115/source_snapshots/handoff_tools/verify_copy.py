"""Recompute all V2 gate/event scores from copied inputs and saved predictions."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

import numpy as np

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    receipt=json.loads((ROOT/'handoff_receipt.json').read_text());bundle=Path(receipt['directory'])
    manifest=json.loads((bundle/'HANDOFF_MANIFEST.json').read_text())
    assert all(sha(bundle/r['file'])==r['sha256'] for r in manifest['files'])
    research=bundle/'research';v1=research/'identity_recovery_20261115';v2=research/'identity_recovery_multiscale_20261115'
    replay=research/'identity_recovery_copy_replay';replay.mkdir(exist_ok=False)
    for directory in ['data','predictions']:(replay/directory).symlink_to(v2/directory,target_is_directory=True)
    for name in ['recovery.py','data_manifest.json']:(replay/name).symlink_to(v2/name)
    algorithm=load('copied_v2',v2/'recovery.py')
    assessment=load('copied_assessment',v1/'assess_controls.py');assessment.ROOT=replay;assessment.recover=algorithm.recover
    stdout=io.StringIO()
    with contextlib.redirect_stdout(stdout):assessment.main()
    (replay/'analysis.log').write_text(stdout.getvalue())
    for name in ['summary.json','cases.json','events.json']:
        assert (v2/'control_results'/name).read_bytes()==(replay/'control_results'/name).read_bytes()
    for original in (v2/'control_results').glob('*.npz'):
        first=np.load(original);second=np.load(replay/'control_results'/original.name)
        assert first.files==second.files and all(np.array_equal(first[k],second[k]) for k in first.files)
    result={'status':'passed','directory':str(bundle),'copied_files_sha256_verified':len(manifest['files']),
            'copied_bytes_verified':manifest['copied_bytes'],'recomputed_v2_sequences':48,
            'summary_case_event_bytes_equal':True,'all_recovered_coordinate_and_mask_arrays_equal':True,
            'native_model_inference_rerun_in_copy':False,
            'scope':'Same workstation: copied V2 inputs/predictions, copied frozen recovery/assessment; no new model inference or cross-machine environment validation.',
            'verification_source_sha256':sha(Path(__file__))}
    target=ROOT/'reproduction_receipt.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(target,v1/target.name)
    additions=[p for p in (replay/'control_results').rglob('*') if p.is_file()]+[replay/'analysis.log',v1/target.name]
    (bundle/'REPRODUCTION_MANIFEST.json').write_text(json.dumps({'verification':result,
        'additions':[{'file':str(p.relative_to(bundle)),'sha256':sha(p)} for p in additions]},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
