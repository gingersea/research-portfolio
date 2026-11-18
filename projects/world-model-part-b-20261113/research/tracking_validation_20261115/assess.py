"""Frozen 2D validation, then paired depth/pose information ablation."""
import argparse
from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fetch_assets import sha
from metrics import lift, score_case

ROOT = Path(__file__).resolve().parent
CENTERS = [0, 5]


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_predictions(inputs, setup, original, foreground):
    video = inputs['video']
    legacy_xy = np.full((len(video), 2, 2), np.nan)
    legacy_vis = np.zeros((len(video), 2), dtype=bool)
    for t, frame in enumerate(video):
        for obj, color in enumerate(setup['colors_rgb_left_right']):
            found = original.component(frame, color)
            if found is not None:
                legacy_xy[t, obj] = found['center']
                legacy_vis[t, obj] = found['complete'] and .75 <= found['area']/setup['expected_areas'][obj] <= 1.25
    tracks = foreground.track_observed(video, setup, video[0])
    foreground_xy = np.array([[item['center'] if item['center'] is not None else [np.nan, np.nan] for item in row] for row in tracks])
    foreground_vis = np.array([[item['eligible'] for item in row] for row in tracks])
    return {'legacy_hue': (legacy_xy, legacy_vis), 'foreground_v1': (foreground_xy, foreground_vis)}


def group_scores(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['split'], row['condition'], row['method'])].append(row)
    output = []
    for (split, condition, method), cases in sorted(grouped.items()):
        count_names = ['evaluated_samples', 'identifiable_samples', 'visible_gt_count', 'invisible_gt_count',
                       'declared_identifiable_count', 'true_visible_claim_count', 'false_visible_count',
                       'pck3_visible_count', 'visible_missing_location_count', 'wrong_other_count',
                       'id_association_transitions', 'ambiguous_sample_count', 'unsupported_identity_count']
        group = {'split': split, 'condition': condition, 'method': method, 'sequences': len(cases)}
        group.update({name: sum(row[name] for row in cases) for name in count_names})
        for name, numerator, denominator in [
            ('visible_precision','true_visible_claim_count','declared_identifiable_count'),
            ('visible_recall','true_visible_claim_count','visible_gt_count'),
            ('pck3_visible','pck3_visible_count','visible_gt_count'),
            ('false_visible_rate','false_visible_count','invisible_gt_count'),
            ('wrong_other_claim_fraction','wrong_other_count','declared_identifiable_count'),
            ('unsupported_identity_fraction','unsupported_identity_count','ambiguous_sample_count')]:
            group[name] = group[numerator]/group[denominator] if group[denominator] else None
        checks = {
            'location': group['pck3_visible'] is not None and group['pck3_visible'] >= .90,
            'visibility_precision': group['visible_precision'] is not None and group['visible_precision'] >= .99,
            'visibility_recall': group['visible_recall'] is not None and group['visible_recall'] >= .90,
            'false_visibility': group['false_visible_rate'] is None or group['false_visible_rate'] <= .01,
            'wrong_identity': group['wrong_other_claim_fraction'] is None or group['wrong_other_claim_fraction'] <= .01,
            'unsupported_identity': group['unsupported_identity_fraction'] is None or group['unsupported_identity_fraction'] <= .01}
        group['failed_gates'] = [key for key, passed in checks.items() if not passed]
        group['all_gates_passed'] = all(checks.values())
        output.append(group)
    return output


def geometry_case(xy, declared, truth):
    gt_xy, gt_xyz = truth['xy'][:, CENTERS], truth['xyz_world'][:, CENTERS]
    known_vis = truth['visible'][:, CENTERS] & truth['identifiable'][:, CENTERS]
    known_vis[0] = False
    finite = np.isfinite(xy).all(axis=-1)
    inside = finite & (xy[..., 0] >= 0) & (xy[..., 0] <= truth['depth'].shape[2]-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= truth['depth'].shape[1]-1)
    sample_xy = np.rint(np.nan_to_num(xy, nan=0)).astype(int)
    sample_xy[..., 0] = np.clip(sample_xy[..., 0], 0, truth['depth'].shape[2]-1)
    sample_xy[..., 1] = np.clip(sample_xy[..., 1], 0, truth['depth'].shape[1]-1)
    sampled = truth['depth'][np.arange(len(xy))[:, None], sample_xy[..., 1], sample_xy[..., 0]]
    temporal_scale = 1+.1*np.sin(np.linspace(0, 2*np.pi, len(xy)))[:, None]
    variants = {'constant_depth_4m': (np.full_like(sampled, 4.), truth['camera_centers']),
                'scene_gt_depth': (sampled, truth['camera_centers']),
                'gt_depth_temporal_10percent': (sampled*temporal_scale, truth['camera_centers']),
                'gt_depth_scale_bias_20percent': (sampled*1.2, truth['camera_centers']),
                'gt_depth_no_camera_compensation': (sampled, np.zeros_like(truth['camera_centers']))}
    rows = []
    for variant, (depth, camera) in variants.items():
        world = lift(xy, depth, truth['intrinsics'], camera)
        error = np.linalg.norm(world-gt_xyz, axis=-1)
        projected = (world-camera[:, None]) @ truth['intrinsics'].T
        reprojection = projected[..., :2]/projected[..., 2:]
        projection_difference = float(np.max(np.abs(reprojection[inside]-xy[inside]))) if inside.any() else None
        for policy, requested in [('all_gt_visible', known_vis), ('declared_visible', known_vis & declared)]:
            mask = requested & inside
            values = error[mask]
            rows.append({'variant': variant, 'policy': policy, 'requested_samples': int(requested.sum()),
                         'evaluated_samples': int(mask.sum()), 'unprojectable_samples': int((requested & ~inside).sum()),
                         'world_epe_m': float(values.mean()) if len(values) else None,
                         'world_median_error_m': float(np.median(values)) if len(values) else None,
                         'projection_change_from_2d_px': projection_difference})
    # Independent exact-query check of the renderer and geometric code.
    gt_pixels = np.rint(gt_xy).astype(int)
    exact_depth = truth['depth'][np.arange(len(xy))[:, None], gt_pixels[..., 1], gt_pixels[..., 0]]
    exact_world = lift(gt_xy, exact_depth, truth['intrinsics'], truth['camera_centers'])
    exact_error = float(np.linalg.norm(exact_world-gt_xyz, axis=-1)[known_vis].max())
    assert exact_error < 1e-5, exact_error
    return rows, exact_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='observer_v2')
    parser.add_argument('--data-manifest', default='data_manifest.json')
    parser.add_argument('--snapshot', default='observer_v2')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(exist_ok=False)
    (output/'baseline_predictions').mkdir()
    manifest = json.loads((ROOT/args.data_manifest).read_text())
    frozen = json.loads((ROOT/'source_snapshots'/args.snapshot/'manifest.json').read_text())
    assert all(sha(ROOT/name) == digest for name, digest in frozen['sha256'].items())
    assert all(sha(ROOT.parent.parent/name) == digest for name, digest in frozen['read_only_baseline_sha256'].items())
    original = module_from(ROOT.parent/'principia_i2v_20261115/evaluate.py', 'original_hue')
    foreground = module_from(ROOT.parent/'tracking_audit_20261115/candidate.py', 'foreground_v1')
    rows, auxiliary, cache = [], [], []
    for case in manifest['cases']:
        assert sha(ROOT/case['input']) == case['input_sha256'] and sha(ROOT/case['truth']) == case['truth_sha256']
        inputs = np.load(ROOT/case['input'], allow_pickle=False)
        truth = np.load(ROOT/case['truth'], allow_pickle=False)
        setup = json.loads(((ROOT/case['input']).parent/'setup.json').read_text())
        base = baseline_predictions(inputs, setup, original, foreground)
        prediction_dir = ROOT/'predictions'/args.run_name/case['split']
        receipt = json.loads((prediction_dir/f'{case["case"]}.json').read_text())
        assert receipt['status'] == 'ok' and not receipt['future_truth_read']
        assert receipt['runner_sha256'] == frozen['sha256']['run_cotracker.py']
        assert receipt['data_manifest_sha256'] == sha(ROOT/args.data_manifest)
        prediction_path = prediction_dir/f'{case["case"]}.npz'
        assert sha(prediction_path) == receipt['prediction_sha256']
        prediction = np.load(prediction_path, allow_pickle=False)
        models = dict(base)
        for method, key in [('cotracker_official','official_visible'), ('cotracker_vc09','conservative_visible')]:
            models[method] = (prediction['xy'][:, CENTERS], prediction[key][:, CENTERS])
            valid_aux = truth['visible'] & truth['identifiable']
            valid_aux[0] = False
            errors = np.linalg.norm(prediction['xy']-truth['xy'], axis=-1)
            auxiliary.append({'case': case['case'], 'split': case['split'], 'condition': case['condition'], 'method': method,
                              'queries': 10, 'visible_identifiable_samples': int(valid_aux.sum()),
                              'pck3_visible': float((errors[valid_aux] <= 3).mean()), 'mean_epe_px': float(errors[valid_aux].mean())})
        for method, (xy, visible) in models.items():
            score = score_case(xy, visible, truth['xy'][:, CENTERS], truth['visible'][:, CENTERS], truth['identifiable'][:, CENTERS])
            rows.append({'case': case['case'], 'condition': case['condition'], 'split': case['split'], 'method': method, **score})
            if method in base:
                np.savez_compressed(output/'baseline_predictions'/f'{case["case"]}_{method}.npz', xy=xy, visible=visible)
        cache.append((case, prediction['xy'][:, CENTERS], prediction['official_visible'][:, CENTERS], dict(truth)))
    grouped = group_scores(rows)
    gates = {method: {'heldout_conditions': len(subset), 'passed_conditions': sum(row['all_gates_passed'] for row in subset),
                       'production_ready': bool(subset) and all(row['all_gates_passed'] for row in subset)}
             for method in ['legacy_hue','foreground_v1','cotracker_official','cotracker_vc09']
             for subset in [[row for row in grouped if row['split']=='heldout' and row['method']==method]]}
    (output/'two_d.json').write_text(json.dumps({'case_scores': rows, 'condition_scores': grouped, 'gates': gates,
                                              'auxiliary_10point_scores': auxiliary}, indent=2, allow_nan=False)+'\n')
    with (output/'two_d_cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Complete 2D validation first, then compare geometric information on identical XY.
    geometry, exact_errors = [], []
    for case, xy, visible, truth in cache:
        measured, exact_error = geometry_case(xy, visible, truth)
        geometry.extend({'case': case['case'], 'condition': case['condition'], 'split': case['split'], **row} for row in measured)
        exact_errors.append(exact_error)
    (output/'geometry.json').write_text(json.dumps({'rows': geometry, 'exact_query_max_error_m': max(exact_errors),
                                                 'scope': 'Known-depth/pose information ablation on unchanged CoTracker XY; no learned 3D tracker or real-depth inference.'}, indent=2, allow_nan=False)+'\n')
    with (output/'geometry_cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(geometry[0])); writer.writeheader(); writer.writerows(geometry)
    print(json.dumps({'successful_native_inference_sequences': len(cache), 'case_method_rows': len(rows), 'gates': gates,
                      'exact_query_geometry_max_error_m': max(exact_errors)}, indent=2))


if __name__ == '__main__':
    main()
