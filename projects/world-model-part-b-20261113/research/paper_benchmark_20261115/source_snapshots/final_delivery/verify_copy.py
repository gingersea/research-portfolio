"""Recompute both benchmark analyses from the copied labels/predictions."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys

from fetch import sha

ROOT=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    receipt=json.loads((ROOT/'handoff_receipt.json').read_text());bundle=Path(receipt['directory'])
    manifest=json.loads((bundle/'HANDOFF_MANIFEST.json').read_text())
    assert all(sha(bundle/r['file'])==r['sha256'] for r in manifest['files'])
    copied=bundle/'research/paper_benchmark_20261115'
    official=load('copied_official_numpy',copied/'official_numpy.py')
    assess=load('copied_assess',copied/'assess.py');assess.ROOT=copied;assess.compute_tapvid_metrics=official.compute_tapvid_metrics
    prior=sys.argv;sys.argv=[str(copied/'assess.py'),'--output','recomputed_davis']
    stdout=io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):assess.main()
    finally:sys.argv=prior
    for name in ['summary.json','per_video.json','reappearance_events.json']:
        assert (copied/'results'/name).read_bytes()==(copied/'recomputed_davis'/name).read_bytes()
    scratch=bundle/'research/paper_benchmark_3d_replay';scratch.mkdir(exist_ok=False)
    for name in ['pilot3d','pilot3d_predictions']:(scratch/name).symlink_to(copied/name,target_is_directory=True)
    (scratch/'pilot3d_manifest.json').symlink_to(copied/'pilot3d_manifest.json')
    assess3=load('copied_assess_3d',copied/'assess_3d.py');assess3.ROOT=scratch
    with contextlib.redirect_stdout(stdout):assess3.main()
    for name in ['summary.json','per_case.json']:
        assert (copied/'pilot3d_results'/name).read_bytes()==(scratch/'pilot3d_results'/name).read_bytes()
    (scratch/'reproduction.log').write_text(stdout.getvalue())
    result={'status':'passed','directory':str(bundle),'copied_files_sha256_verified':len(manifest['files']),
            'copied_bytes_verified':manifest['copied_bytes'],'recomputed_human_labeled_videos':30,
            'recomputed_geometry_pilot_videos':3,'all_summary_case_event_outputs_bytes_equal':True,
            'native_inference_rerun_in_copy':False,'scope':'Same workstation, copied inputs/labels/predictions and copied official scorer/assessment; no cross-machine environment validation.',
            'verification_source_sha256':sha(Path(__file__))}
    target=ROOT/'reproduction_receipt.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(target,copied/target.name)
    additions=[p for p in (copied/'recomputed_davis').rglob('*') if p.is_file()]+[p for p in (scratch/'pilot3d_results').rglob('*') if p.is_file()]+[scratch/'reproduction.log',copied/target.name]
    (bundle/'REPRODUCTION_MANIFEST.json').write_text(json.dumps({'verification':result,
        'additions':[{'file':str(p.relative_to(bundle)),'sha256':sha(p)} for p in additions]},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
