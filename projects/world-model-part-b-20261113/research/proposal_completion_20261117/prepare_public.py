"""Existing official labels, all DAVIS and full-length PStudio, truth separate."""
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha
from physcheck.relations.backends import validate_queries


def entry(path):
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def main():
    manifest = json.loads((PROJECT / 'research/integrated_matrix_20261115/manifest.json').read_text())
    manifest['cases'] = [c for c in manifest['cases'] if c['domain'] == 'davis']
    for case in manifest['cases']:
        data = np.load(case['files']['input']['path'])
        validate_queries(data['queries_tyx'], (case['frames'], 256, 256, 3))
    paper = PROJECT / 'research/paper_benchmark_20261115'
    prior = {c['case']: c for c in json.loads((paper / 'pilot3d_manifest.json').read_text())['cases']}
    for source in json.loads((paper / 'pilot3d_assets.json').read_text())['cases']:
        raw_path = Path(source['local'])
        assert sha(raw_path) == source['sha256']
        raw = np.load(raw_path, allow_pickle=False)
        name = Path(source['name']).stem
        ids = prior[name]['selected_point_ids']
        rgb = np.stack([np.asarray(Image.open(io.BytesIO(bytes(s))).convert('RGB')) for s in raw['images_jpeg_bytes']])
        query = raw['queries_xyt'][ids][:, [2, 1, 0]].astype(np.float32)
        validate_queries(query, rgb.shape)
        fx, fy, cx, cy = raw['fx_fy_cx_cy']
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        xyz, visible = raw['tracks_XYZ'][:, ids], raw['visibility'][:, ids]
        projected = xyz @ k.T
        xy = projected[..., :2] / projected[..., 2:]
        assert np.max(np.linalg.norm(xy[query[:, 0].astype(int), np.arange(8)] - query[:, [2, 1]], axis=-1)) < 1.
        folder = ROOT / 'pstudio_full' / name
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / 'input.npz', video=rgb, queries_tyx=query, intrinsics=k)
        np.savez_compressed(folder / 'truth.npz', tracks_xyz=xyz, tracks_xy=xy, visible=visible)
        manifest['cases'].append({'id': 'pstudio_' + name, 'domain': 'pstudio', 'frames': len(rgb),
             'selected_point_ids': ids, 'source_sha256': source['sha256'], 'files': {
                 'input': entry(folder / 'input.npz'), 'truth': entry(folder / 'truth.npz')}})
    manifest['selection'] = 'all30DAVIS + same3PStudio full150frames/same8queries; no score-based selection'
    save(ROOT / 'public_manifest.json', manifest)
    snapshot = ROOT / 'source_snapshots/pre_public_inference'
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in (PROJECT / 'benchmark/physcheck/relations').glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = sha(path)
    save(snapshot / 'receipt.json', {'sources': hashes, 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
         'manifest_sha256': sha(ROOT / 'public_manifest.json'), 'new_unseen_data_claim': False,
         'Davis_queries': 650, 'PStudio_frames_each': 150, 'GT_inference_inputs': False})
    print(json.dumps({'videos': 33, 'davis_queries': 650, 'pstudio_total_frames': 450}))


if __name__ == '__main__':
    main()
