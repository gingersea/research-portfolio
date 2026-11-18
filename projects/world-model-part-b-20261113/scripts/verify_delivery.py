"""Verify the portable published files and frozen video evidence, without writes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest or root/'delivery_manifest.json'
    if not manifest_path.is_file():
        parser.error(f'manifest not found: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['repository_files'].items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'File checksum mismatch: {name}')
    import av
    videos = json.loads((root/'data/video_manifest.json').read_text())
    rows = [json.loads(line) for line in (root/'benchmark/results/raw/part_a.jsonl').read_text().splitlines()]
    keyed = {(r['model'],r['prompt_id'],r['seed']):r for r in rows}
    assert len(videos) == len(keyed) == len(rows) == 24
    total = 0
    for record in videos:
        video = root/record['path']
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        row = keyed[(record['model'],record['prompt_id'],record['seed'])]
        meta = json.loads(video.with_suffix('.json').read_text())
        assert digest == record['sha256'] == row['video_sha256'] == meta['video_sha256']
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            dimensions = [stream.width, stream.height]
            count = sum(1 for _ in container.decode(video=0))
        assert fps == row['fps'] == meta['fps']
        assert count == meta['frames'] and dimensions == meta['resolution']
        total += count
    assert total == 1560
    digest = hashlib.sha256()
    with tarfile.open(root/'data/frozen_physcheck_source.tar.gz') as tar:
        for member in sorted(tar.getmembers(), key=lambda m:m.name):
            digest.update(member.name.encode())
            digest.update(tar.extractfile(member).read())
    assert {r['analysis_source_sha256'] for r in rows} == {digest.hexdigest()}
    training = root/'benchmark/results/training_pilot'
    metrics = json.loads((training/'metrics.json').read_text())
    assert hashlib.sha256((training/'best_model.pt').read_bytes()).hexdigest() == metrics['model_sha256']
    assert hashlib.sha256((root/'benchmark/training_pilot.py').read_bytes()).hexdigest() == metrics['source_sha256']
    multi = root/'research/multiscale_20261109T191120Z/results'
    frozen = json.loads((multi/'checkpoint_freeze.json').read_text())
    assert len(frozen) == 6
    for name, expected in frozen.items():
        assert hashlib.sha256((multi/name).read_bytes()).hexdigest() == expected
    print(json.dumps({'passed':True,'repository_files_verified':len(manifest['repository_files']),
                      'decoded_videos':24,'decoded_frames':total,'new_checkpoints':6,
                      'legacy_analysis_source_sha256':digest.hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
