"""Versioned local handoff with exact before/after sources and copied-file hashes."""
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil

from fetch_tapip import sha

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
SHARED=Path('/data/AI_Research/shared')


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    frozen=json.loads((ROOT/'pre_inference_freeze.json').read_text())
    snapshots=ROOT/'source_snapshots';snapshots.mkdir(exist_ok=False)
    before=snapshots/'pre_inference';before.mkdir()
    for name,digest in frozen['files'].items():
        source=ROOT/name
        if sha(source)!=digest:
            assert name=='run_model.py'
            source=ROOT/'run_model.before_hub_redirect.py'
        assert sha(source)==digest
        shutil.copy2(source,before/name)
    after=snapshots/'final';after.mkdir()
    patches=[]
    for source in sorted(ROOT.iterdir()):
        if source.suffix not in ['.py','.md','.json']:continue
        shutil.copy2(source,after/source.name)
        if source.suffix in ['.py','.md']:
            baseline=(before/source.name).read_text().splitlines(keepends=True) if (before/source.name).exists() else []
            patches.extend(difflib.unified_diff(baseline,source.read_text().splitlines(keepends=True),
                           fromfile='pre_inference/'+source.name if baseline else '/dev/null',
                           tofile='final/'+source.name))
    (snapshots/'exact_diff.patch').write_text(''.join(patches))
    (snapshots/'manifest.json').write_text(json.dumps({
        'before':{p.name:sha(p) for p in before.iterdir()},
        'after':{p.name:sha(p) for p in after.iterdir()},
        'scope':'New research only; existing baseline source identities verified separately.'},indent=2)+'\n')
    destination=SHARED/'handoff'/f'world_model_observation_gate_20261115_v1_{stamp}'
    destination.mkdir(parents=True,exist_ok=False)
    copied=[]

    def copy(source,relative):
        source=Path(source);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert sha(source)==sha(target)
            return
        shutil.copy2(source,target)
        digest=sha(source)
        assert sha(target)==digest
        copied.append({'file':str(relative),'sha256':digest,'bytes':target.stat().st_size})

    for source in ROOT.rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:
            copy(source,source.relative_to(PROJECT))
    old=ROOT.parent/'tracking_validation_20261115'
    for name in ['assets.json','data_manifest_v2.json','render_controls.py','metrics.py','PROTOCOL.md']:
        copy(old/name,(old/name).relative_to(PROJECT))
    for sub in ['data_v2','predictions/dataset_v2','source_snapshots/dataset_v2']:
        for source in (old/sub).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                copy(source,source.relative_to(PROJECT))
    helper=ROOT.parent/'physics_methods_20261114/fetch_ranges.py'
    copy(helper,helper.relative_to(PROJECT))
    copy(PROJECT/'docs/OBSERVATION_GATE_2026-11-15.zh.md',Path('docs/OBSERVATION_GATE_2026-11-15.zh.md'))
    plan=SHARED/'research_plans/world_model_observation_gate_20261115.md'
    copy(plan,Path('research_plan.md'))
    co=json.loads((old/'assets.json').read_text())
    tap=json.loads((ROOT/'tapip_assets.json').read_text())
    tapsource=json.loads((ROOT/'tapip_source.json').read_text())
    for method,assets,sources in [('cotracker',co,co['source_files']),('tapip3d',tap,tapsource['files'])]:
        for row in sources:
            source=Path(assets['source_directory'])/row['file']
            assert sha(source)==row['sha256']
            copy(source,Path('assets')/method/'source'/row['file'])
        copy(assets['checkpoint'],Path('assets')/method/Path(assets['checkpoint']).name)
    extension=Path(tap['source_directory'])/'third_party/pointops2/pointops2_cuda.cpython-312-x86_64-linux-gnu.so'
    copy(extension,Path('assets/tapip3d')/extension.name)
    for wheel in (SHARED/'observation_gate_20261115/wheels').glob('*.whl'):
        copy(wheel,Path('assets/environment_wheels')/wheel.name)
    receipt={'directory':str(destination),'copied_files':len(copied),'copied_bytes':sum(r['bytes'] for r in copied),
             'copy_sha256_verified':True,'files':copied,
             'scope':'Local handoff; ML base environment and CUDA toolchain remain external. No public upload.',
             'asset_path_note':'Copied receipts retain original absolute asset paths. Repoint them in a new study copy for another machine.'}
    (destination/'HANDOFF_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'handoff_receipt.json').write_text(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2)+'\n')
    shutil.copy2(ROOT/'handoff_receipt.json',destination/ROOT.relative_to(PROJECT)/'handoff_receipt.json')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))


if __name__=='__main__':main()
