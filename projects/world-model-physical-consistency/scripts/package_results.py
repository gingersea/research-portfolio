"""Copy frozen local evidence and build the GitHub delivery artifact.

This exports existing research artifacts; it never reruns generation or rewrites
historical records. Replays are redundant and omitted from the archive.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_once(source, destination):
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f'Conflicting existing export: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists():
        parser.error('--archive must be a new file')
    source = args.source.resolve()
    rows = [json.loads(line) for line in (source/'benchmark/results/raw/part_a.jsonl').read_text().splitlines()]
    assert len(rows) == 24
    for folder in ['benchmark/results/raw', 'benchmark/results/training_pilot']:
        for path in sorted((source/folder).rglob('*')):
            if path.is_file():
                copy_once(path, ROOT/path.relative_to(source))
    video_records = []
    for row in rows:
        video = Path(row['video_path'])
        assert sha(video) == row['video_sha256']
        relative = Path('data/generated_videos')/row['model']/row['prompt_id']/video.name
        copy_once(video, ROOT/relative)
        copy_once(video.with_suffix('.json'), (ROOT/relative).with_suffix('.json'))
        video_records.append({'model':row['model'], 'prompt_id':row['prompt_id'], 'seed':row['seed'],
                              'path':relative.as_posix(), 'sha256':sha(video), 'fps':row['fps']})
    # Preserve the exact historical analysis source instead of pretending that
    # the later frame-sampling fix was used in the original audit.
    frozen = ROOT/'data/frozen_physcheck_source.tar.gz'
    if not frozen.exists():
        with tarfile.open(frozen, 'w:gz') as tar:
            for path in sorted((source/'benchmark/physcheck').rglob('*')):
                rel = path.relative_to(source/'benchmark/physcheck')
                if path.is_file() and path.suffix in {'.py', '.json'} and not set(rel.parts)&{'artifacts','models_cache','__pycache__'}:
                    payload = path.read_bytes()
                    info = tarfile.TarInfo(rel.as_posix())
                    info.size = len(payload)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(payload))
    (ROOT/'data/video_manifest.json').write_text(json.dumps(video_records, indent=2)+'\n')
    for name in ['part_a_halfway_results.md','progress-2026-11-06.zh.md']:
        copy_once(source/'report'/name, ROOT/'report'/name)
    for path in (source/'report/milestone').iterdir():
        if path.is_file() and path.suffix in {'.py','.md','.tex','.bib','.json','.txt'}:
            copy_once(path, ROOT/'report/milestone'/path.name)
    shared = Path('/data/AI_Research/shared')
    for path in shared.joinpath('research_plans').glob('world_model_*.md'):
        copy_once(path, ROOT/'docs/protocols'/path.name)
    copy_once(shared/'handoff/world_model_cleanup_20261109T144046Z/video_code_snapshot.json', ROOT/'docs/provenance/video_code_import.json')
    env = {'python':sys.version, 'torch_runtime':__import__('torch').__version__,
           'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','opencv-python-headless','imageio','av','matplotlib','pytest']}}
    (ROOT/'environment-verified.json').write_text(json.dumps(env, indent=2)+'\n')
    # Canonical source, evidence and full derived image outputs, with no local
    # model cache, repository metadata or duplicate verification replays.
    payload_paths = []
    for folder in ['benchmark','data','docs','report','research','scripts','tests']:
        for path in (ROOT/folder).rglob('*'):
            rel = path.relative_to(ROOT)
            if not path.is_file() or set(rel.parts)&{'__pycache__','.pytest_cache','artifacts','models_cache','replay_test','replay_verified'}:
                continue
            payload_paths.append(path)
    payload_paths += [ROOT/name for name in ['README.md','.gitignore','pytest.ini','requirements-analysis.txt','environment-verified.json']]
    archive_files = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(payload_paths)}
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'w:gz') as tar:
        for path in sorted(payload_paths):
            tar.add(path, arcname='world-model-physical-consistency/'+path.relative_to(ROOT).as_posix())
    candidates = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'], cwd=ROOT).decode().split('\0')
    repo_files = {name:sha(ROOT/name) for name in sorted(set(candidates)) if name and (ROOT/name).is_file() and name != 'delivery_manifest.json'}
    manifest = {'release_tag':'video-results-2026-11-12',
                'archive':{'name':args.archive.name,'sha256':sha(args.archive),'bytes':args.archive.stat().st_size,'files':archive_files},
                'repository_files':repo_files, 'n_videos':24, 'n_frames':1560,
                'scope':'frozen artifacts and derived results; no new experiments',
                'excluded':'model caches, bytecode, duplicate replays; historical checksum files may describe those local replays'}
    (ROOT/'delivery_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'archive':str(args.archive),'archive_bytes':args.archive.stat().st_size,
                      'archive_files':len(archive_files),'repository_files':len(repo_files)}, indent=2))


if __name__ == '__main__':
    main()
