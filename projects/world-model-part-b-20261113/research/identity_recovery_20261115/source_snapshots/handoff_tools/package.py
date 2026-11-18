"""Versioned local delivery, author assets, exact source diffs and SHA receipts."""
from datetime import datetime,timezone
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent
V2=ROOT.parent/'identity_recovery_multiscale_20261115'
PROJECT=ROOT.parent.parent
SHARED=Path('/data/AI_Research/shared')


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=SHARED/'handoff'/f'world_model_identity_recovery_20261115_v1_{stamp}'
    destination.mkdir(parents=True,exist_ok=False)
    for directory in [ROOT,V2]:
        subprocess.run([sys.executable,str(directory/'freeze.py'),'handoff_tools'],check=True,capture_output=True,text=True)
        diff=[]
        for path in sorted(directory.glob('*.py')):
            prefix=path.stem+'.before_'
            before=next(iter(sorted(directory.glob(prefix+'*.py'))),None)
            previous=before.read_text().splitlines(keepends=True) if before else []
            diff.extend(difflib.unified_diff(previous,path.read_text().splitlines(keepends=True),
                fromfile=before.name if before else '/dev/null',tofile=path.name))
        (directory/'source_snapshots/exact_source_changes.patch').write_text(''.join(diff))
    (ROOT/'source_snapshots/recovery_v1_to_v2.patch').write_text(''.join(difflib.unified_diff(
        (ROOT/'recovery.py').read_text().splitlines(keepends=True),(V2/'recovery.py').read_text().splitlines(keepends=True),
        fromfile='v1/recovery.py',tofile='v2/recovery.py')))
    receipt={'directory':str(destination),'manifest':str(destination/'HANDOFF_MANIFEST.json'),
             'scope':'Local versioned research handoff, no public upload.',
             'human_labels':'Not available; user authorized AI reference and pending human annotation package.',
             'portability':'Existing isolated Python/CUDA environment external. Original absolute asset paths retained in copied receipts; reconfigure paths in a new copy for another machine.'}
    assert not (ROOT/'handoff_receipt.json').exists()
    (ROOT/'handoff_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    files=[]

    def copy(source,relative):
        source=Path(source);target=destination/relative
        if target.exists():assert sha(source)==sha(target);return
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        digest=sha(source);assert sha(target)==digest
        files.append({'file':str(relative),'sha256':digest,'bytes':target.stat().st_size})

    for directory in [ROOT,V2]:
        for source in directory.rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and '.pytest_cache' not in source.parts:
                copy(source,source.relative_to(PROJECT))
    for directory in ['observation_gate_20261115','tracking_validation_20261115']:
        for source in (ROOT.parent/directory).iterdir():
            if source.suffix in ['.py','.json','.md']:copy(source,source.relative_to(PROJECT))
    original=ROOT.parent/'principia_i2v_20261115'
    for relative in ['input_manifest.json','PROTOCOL.md','REPORT.zh.md','inputs/A/setup.json','inputs/B/setup.json']:
        copy(original/relative,(original/relative).relative_to(PROJECT))
    for case in json.loads((ROOT/'real_manifest.json').read_text()):
        source=Path(case['source']);copy(source,source.relative_to(PROJECT))
        for suffix in ['.mp4','.json']:
            companion=source.parent/(source.stem.removesuffix('_frames')+suffix)
            copy(companion,companion.relative_to(PROJECT))
    helper=ROOT.parent/'physics_methods_20261114/fetch_ranges.py';copy(helper,helper.relative_to(PROJECT))
    copy(PROJECT/'docs/IDENTITY_RECOVERY_2026-11-15.zh.md',Path('docs/IDENTITY_RECOVERY_2026-11-15.zh.md'))
    for name in ['world_model_identity_recovery_20261115.md','world_model_identity_recovery_v2_20261115.md']:
        copy(SHARED/'research_plans'/name,Path('plans')/name)
    co=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    tap=json.loads((ROOT.parent/'observation_gate_20261115/tapip_assets.json').read_text())
    tapfiles=json.loads((ROOT.parent/'observation_gate_20261115/tapip_source.json').read_text())['files']
    depth=json.loads((ROOT/'depth_assets.json').read_text())
    for name,assets,sources in [('cotracker',co,co['source_files']),('tapip3d',tap,tapfiles),('depth_anything_v2',depth,depth['files'])]:
        for row in sources:
            source=Path(assets['source_directory'])/row['file'];assert sha(source)==row['sha256']
            copy(source,Path('assets')/name/'source'/row['file'])
        copy(assets['checkpoint'],Path('assets')/name/Path(assets['checkpoint']).name)
    extension=Path(tap['source_directory'])/'third_party/pointops2/pointops2_cuda.cpython-312-x86_64-linux-gnu.so'
    copy(extension,Path('assets/tapip3d/source/third_party/pointops2')/extension.name)
    manifest={'status':'copy_sha256_verified','files':files,'copied_files':len(files),'copied_bytes':sum(r['bytes'] for r in files),**receipt}
    (destination/'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':main()
