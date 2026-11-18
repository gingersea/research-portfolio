"""Complete local experiment delivery, preserving failures, versions and assets."""
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import shutil

from fetch_assets import sha

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent


def main():
    receipt_path = ROOT/'handoff_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Delivery exists; do not replace')
    verification = json.loads((ROOT/'verification_figures_v2.json').read_text())
    assert verification['final_native_sequences']==30 and verification['unique_final_video_pixel_hashes']==30
    assert not verification['gates']['cotracker_official']['production_ready']
    results = json.loads((ROOT/'results_v2/two_d.json').read_text())
    row = next(row for row in results['condition_scores'] if row['condition']=='identity_replacement' and row['split']=='heldout' and row['method']=='cotracker_official')
    assert row['false_visible_count']==35 and row['invisible_gt_count']==50
    links = []
    for source in [ROOT/'REPORT.zh.md', PROJECT/'docs/TRACKING_VALIDATION_2026-11-15.zh.md']:
        for link in re.findall(r'\]\(([^)]+)\)',source.read_text()):
            if not link.startswith(('http:','https:','#')):
                path = (source.parent/link.split('#')[0]).resolve()
                assert path.exists(), (source,link)
                links.append(str(path))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path('/data/AI_Research/shared/handoff')/f'world_model_tracking_validation_20261115_v1_{stamp}'
    target.mkdir(parents=True,exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__','*.pyc')
    original_equal = 0
    for study in [ROOT,ROOT.parent/'tracking_audit_20261115',ROOT.parent/'principia_i2v_20261115']:
        destination = target/'research'/study.name
        shutil.copytree(study,destination,ignore=ignore)
        for path in study.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                assert sha(path)==sha(destination/path.relative_to(study))
                original_equal += 1
    dependencies = ['research/principia_gravity_20261114/measure.py',
                    'research/principia_gravity_20261114/source_snapshots/heldout_v2/manifest.json',
                    'research/physics_methods_20261114/fetch_ranges.py',
                    'docs/TRACKING_VALIDATION_2026-11-15.zh.md']
    for relative in dependencies:
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/relative,destination)
    assets = json.loads((ROOT/'assets.json').read_text())
    for item in assets['source_files']:
        source=Path(assets['source_directory'])/item['file']
        assert sha(source)==item['sha256']
        destination=target/'external_assets/co-tracker'/item['file']
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    checkpoint=Path(assets['checkpoint'])
    assert sha(checkpoint)==assets['checkpoint_sha256']
    shutil.copy2(checkpoint,target/'external_assets/scaled_offline.pth')
    assert sha(target/'external_assets/scaled_offline.pth')==assets['checkpoint_sha256']
    shutil.copy2('/data/AI_Research/shared/research_plans/world_model_tracking_validation_20261115.md',target/'RESEARCH_PLAN.md')
    history=[]
    snapshots=sorted((ROOT/'source_snapshots').glob('*/manifest.json'),key=lambda path:json.loads(path.read_text())['utc'])
    for before,after in zip(snapshots,snapshots[1:]):
        names=sorted(set(json.loads(before.read_text())['sha256'])|set(json.loads(after.read_text())['sha256']))
        for name in names:
            left,right=before.parent/name,after.parent/name
            history.extend(difflib.unified_diff(left.read_text().splitlines(keepends=True) if left.exists() else [],
                                               right.read_text().splitlines(keepends=True) if right.exists() else [],
                                               fromfile=f'{before.parent.name}/{name}',tofile=f'{after.parent.name}/{name}'))
    (target/'SOURCE_HISTORY.patch').write_text(''.join(history))
    (target/'HANDOFF.md').write_text(
        '# Persistent tracking validation and geometry ablation\n\n'
        'Read research/tracking_validation_20261115/REPORT.zh.md. Final evidence uses data_v2, '
        'predictions/dataset_v2, results_v2 and figures_v2. Thirty final controls completed; '
        'the application identity/visibility gate FAILED. No physics-score or generation expansion.\n\n'
        'Initial failed observer attempt and v1 data with cross-split duplicates are retained for audit. '
        'v1 results are not valid heldout evidence. Geometry results use known scene depth/camera '
        'on unchanged2D predictions; full learned3D tracking was not run.\n\n'
        'The official CoTracker3 checkpoint(101,890,938bytes), pinned source/license and all control inputs '
        'are included under external_assets and research. Historical predecessor artifacts are included '
        'for baseline/context integrity; their large LTX generator weights are not included. '
        'No conversation transcripts, credentials or raw reasoning. This is local delivery, not publication.\n\n'
        'On this workstation assets.json retains the verified original absolute paths. To relocate, '
        'edit ONLY a new copy of assets.json so source_directory points to external_assets/co-tracker '
        'and checkpoint to external_assets/scaled_offline.pth; preserve repo revisions/hashes. '
        'Use new output directories/run names. Package relocation inference was not claimed tested.\n\n'
        'MANIFEST.sha256.json covers every delivered file except itself; the source study '
        'handoff_receipt.json records its hash. SOURCE_HISTORY.patch records source changes.\n')
    manifest={str(path.relative_to(target)):sha(path) for path in sorted(target.rglob('*')) if path.is_file()}
    manifest_path=target/'MANIFEST.sha256.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    assert all(sha(target/name)==digest for name,digest in manifest.items())
    record={'handoff_directory':str(target),'created_utc':stamp,'manifest_sha256':sha(manifest_path),
            'verified_files':len(manifest),'source_study_files_equal':original_equal,'local_report_links_verified':len(links),
            'bytes':sum((target/name).stat().st_size for name in manifest),'all_copy_hashes_passed':True,
            'native_tracking_checkpoint_included':True,'large_video_generator_weights_included':False,
            'production_gate_passed':False,'full_learned_3d_model_run':False,'public_upload':False}
    receipt_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
