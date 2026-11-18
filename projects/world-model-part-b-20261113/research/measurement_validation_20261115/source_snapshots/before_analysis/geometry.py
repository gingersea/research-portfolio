"""GT-assisted error attribution in evaluator ONLY; no model inference."""
from datetime import datetime, timezone
import json
import sys

import numpy as np

from common import PAPER, ROOT, save, sha


def backproject(xy, depth, k):
    ray = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1) @ np.linalg.inv(k).T
    return ray * depth[..., None]


def sample_depth(depth, xy):
    h, w = depth.shape[1:]
    index = np.rint(xy).astype(int)
    index[..., 0] = index[..., 0].clip(0, w - 1)
    index[..., 1] = index[..., 1].clip(0, h - 1)
    return depth[np.arange(len(depth))[:, None], index[..., 1], index[..., 0]]


def main():
    out = ROOT / 'geometry_results'
    out.mkdir(exist_ok=False)
    rows, bindings = [], []
    for case in json.loads((PAPER / 'pilot3d_manifest.json').read_text())['cases']:
        ip, tp = PAPER / case['input'], PAPER / case['truth']
        pp = PAPER / 'pilot3d_predictions' / (case['case'] + '.npz')
        receipt = json.loads(pp.with_suffix('.json').read_text())
        assert sha(ip) == case['input_sha256'] and sha(tp) == case['truth_sha256']
        assert sha(pp) == receipt['prediction_sha256'] and not receipt['future_truth_read']
        bindings.append({str(p): sha(p) for p in [ip, tp, pp]})
        inputs, truth, pred = np.load(ip), np.load(tp), np.load(pp)
        xyz, xy, k = truth['tracks_xyz'], truth['tracks_xy'], inputs['intrinsics']
        positive = truth['visible'] & (np.arange(case['frames'])[:, None] > inputs['queries_tyx'][None, :, 0])
        z = xyz[..., 2]
        true_ray = xyz / z[..., None]
        assert np.max(np.abs(backproject(xy, z, k) - xyz)) < 1e-5
        depth_true_xy = sample_depth(pred['depth'], xy)
        for prefix in ['co', 'tap']:
            pxyz, pxy = pred[prefix + '_xyz'], pred[prefix + '_xy']
            pz = pxyz[..., 2]
            pred_ray = pxyz / pz[..., None]
            assert np.max(np.abs(backproject(pxy, pz, k) - pxyz)) < 1e-4
            depth_component = (pz - z)[..., None] * true_ray
            ray_component = pz[..., None] * (pred_ray - true_ray)
            residual = np.max(np.abs((pxyz - xyz) - (depth_component + ray_component)))
            assert residual < 1e-5
            alternatives = {'raw': pxyz,
                'oracle_true_z': pred_ray * z[..., None],
                'oracle_true_ray': true_ray * pz[..., None],
                'oracle_xy_da_depth': true_ray * depth_true_xy[..., None]}
            row = {'case': case['case'], 'method': prefix, 'visible_post_query': int(positive.sum()),
                   'decomposition_max_residual_m': float(residual),
                   'oracle_gt_xy_out_of_bounds': int((positive & ((xy[..., 0] < 0) | (xy[..., 0] >= pred['depth'].shape[2]) |
                              (xy[..., 1] < 0) | (xy[..., 1] >= pred['depth'].shape[1]))).sum()),
                   'predicted_depth_absrel': float(np.mean(np.abs(pz - z)[positive] / z[positive])),
                   'da_at_true_xy_absrel': float(np.mean(np.abs(depth_true_xy - z)[positive] / z[positive]))}
            for name, candidate in alternatives.items():
                err = np.linalg.norm(candidate - xyz, axis=-1)
                row[name + '_epe_m'] = float(err[positive].mean())
                row[name + '_within_5cm'] = int((positive & (err < .05)).sum())
            rows.append(row)
    summary = []
    for method in ['co', 'tap']:
        selected = [r for r in rows if r['method'] == method]
        summary.append({'method': method, 'sequences': 3, 'queries': 24,
             **{key: float(np.mean([r[key] for r in selected])) for key in selected[0]
                if key.endswith('_epe_m') or key.endswith('_absrel')}})
    save(out / 'per_video.json', rows)
    save(out / 'summary.json', summary)
    save(out / 'receipt.json', {'utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
         'source_sha256': sha(ROOT / 'geometry.py'), 'inputs': bindings,
         'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'oracle_GT_used_only_in_evaluation': True,
         'neural_inference_rerun': False, 'no_scale_fit': True})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
