"""Versioned local delivery with derived benchmark inputs, labels, source and assets."""
from datetime import datetime,timezone
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fetch import sha

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
SHARED=Path('/data/AI_Research/shared')


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=SHARED/'handoff'/f'world_model_paper_benchmark_20261115_v1_{stamp}'
    destination.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(ROOT/'freeze.py'),'final_delivery'],check=True,capture_output=True,text=True)
    patch=[]
    for path in sorted(ROOT.glob('*.py')):
        options=sorted(ROOT.glob(path.stem+'.before_*.py'),key=lambda p:p.stat().st_mtime)
        before=options[0] if options else ROOT/'source_snapshots/before_davis_inference'/path.name
        previous=before.read_text().splitlines(keepends=True) if before.exists() else []
        patch.extend(difflib.unified_diff(previous,path.read_text().splitlines(keepends=True),
            fromfile=str(before.relative_to(ROOT)) if before.exists() else '/dev/null',tofile=path.name))
    (ROOT/'source_snapshots/exact_source_changes.patch').write_text(''.join(patch))
    receipt={'directory':str(destination),'manifest':str(destination/'HANDOFF_MANIFEST.json'),
             'scope':'Local research delivery, no public upload.',
             'raw_data':'Complete original official ZIP/pickle and minival files remain at shared/paper_benchmark_20261115, bound by dataset_assets.json/pilot3d_assets.json.',
             'included':'All derived inputs, labels, predictions, official scorer/source, results, provenance/attribution and copied model assets.',
             'portability':'Base ML/CUDA environment external; original absolute asset paths retained in receipts; reconfigure paths in a new study copy on another machine.'}
    target=ROOT/'handoff_receipt.json';assert not target.exists();target.write_text(json.dumps(receipt,indent=2)+'\n')
    files=[]

    def copy(source,relative):
        source=Path(source);dest=destination/relative
        if dest.exists():assert sha(source)==sha(dest);return
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        digest=sha(source);assert sha(dest)==digest
        files.append({'file':str(relative),'sha256':digest,'bytes':dest.stat().st_size})

    for source in ROOT.rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and '.pytest_cache' not in source.parts:
            copy(source,source.relative_to(PROJECT))
    copy(PROJECT/'docs/PAPER_BENCHMARK_2026-11-15.zh.md',Path('docs/PAPER_BENCHMARK_2026-11-15.zh.md'))
    copy(SHARED/'research_plans/world_model_paper_benchmark_20261115.md',Path('plans/world_model_paper_benchmark_20261115.md'))
    for relative in ['tracking_validation_20261115/assets.json','observation_gate_20261115/tapip_assets.json',
                     'observation_gate_20261115/tapip_source.json','identity_recovery_20261115/depth_assets.json']:
        copy(ROOT.parent/relative,Path('research')/relative)
    co=json.loads((ROOT.parent/'tracking_validation_20261115/assets.json').read_text())
    tap=json.loads((ROOT.parent/'observation_gate_20261115/tapip_assets.json').read_text())
    da=json.loads((ROOT.parent/'identity_recovery_20261115/depth_assets.json').read_text())
    tapsource=json.loads((ROOT.parent/'observation_gate_20261115/tapip_source.json').read_text())['files']
    for name,assets,sources in [('cotracker',co,co['source_files']),('tapip3d',tap,tapsource),('depth_anything_v2',da,da['files'])]:
        for row in sources:
            source=Path(assets['source_directory'])/row['file'];assert sha(source)==row['sha256']
            copy(source,Path('assets')/name/'source'/row['file'])
        copy(assets['checkpoint'],Path('assets')/name/Path(assets['checkpoint']).name)
    extension=Path(tap['source_directory'])/'third_party/pointops2/pointops2_cuda.cpython-312-x86_64-linux-gnu.so'
    copy(extension,Path('assets/tapip3d/source/third_party/pointops2')/extension.name)
    dataset=json.loads((ROOT/'dataset_assets.json').read_text())
    for row in dataset['files']:
        if row['file'].endswith('.md'):
            copy(Path(dataset['release_directory'])/row['file'],Path('dataset_attribution')/row['file'])
    for row in json.loads((ROOT/'pilot3d_assets.json').read_text())['cases']:
        copy(row['local'],Path('raw_pstudio_pilot')/row['name'])
    dependency=json.loads((ROOT/'resize_dependency.json').read_text());copy(dependency['local'],Path('assets/environment')/Path(dependency['local']).name)
    manifest={'status':'copy_sha256_verified','files':files,'copied_files':len(files),'copied_bytes':sum(r['bytes'] for r in files),**receipt}
    (destination/'HANDOFF_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':main()
