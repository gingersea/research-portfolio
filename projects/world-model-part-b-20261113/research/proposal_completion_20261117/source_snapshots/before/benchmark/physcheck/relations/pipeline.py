"""Single orchestration surface; fixed algorithms and explicit scientific limits."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .contracts import lift_points, module, relation_decision, save, sha, verify_cache, verify_file
from .visual import index_opinions, opaque_id, validate_opinion


def aggregate_rows(rows):
    domains = {}
    for domain in sorted({r['domain'] for r in rows}):
        selected = [r for r in rows if r['domain'] == domain]
        domains[domain] = {'cases': len(selected), 'errors': sum(r['status'] == 'error' for r in selected),
            'verdicts': dict(Counter(r['relation']['verdict'] for r in selected if 'relation' in r))}
        evaluations = [r['evaluation'] for r in selected if r.get('evaluation')]
        metrics = {}
        for method in sorted({m for e in evaluations for m in e}):
            values = [e[method] for e in evaluations if method in e]
            metrics[method] = {'videos': len(values), **{key: float(np.mean([v[key] for v in values]))
                for key in values[0] if key not in ['visible_point_times']}}
        domains[domain]['video_mean_metrics'] = metrics
    return {'total_cases': len(rows), 'errors': sum(r['status'] == 'error' for r in rows),
            'visual_status_counts': dict(Counter(r.get('visual_judge', {}).get('status', 'not_applicable') for r in rows)),
            'i2v_independent_visual_pending': sum(r['domain'] == 'i2v' and not r.get('visual_judge', {}).get('independent_validation', False) for r in rows),
            'by_domain': domains, 'stage_status_counts': dict(Counter(
                name + ':' + stage['status'] for r in rows for name, stage in r.get('stages', {}).items()))}


def cached(case, name):
    entry = case['files'][name]
    path = verify_file(entry)
    receipt = json.loads(verify_file(case['files'][name + '_receipt']).read_text())
    verify_cache(receipt, case['files']['input']['sha256'], entry['sha256'])
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files}
    return values


def rgb_sheet(video, folder, opaque_id):
    """All frames, with original frame indices; no trajectories/model names/GT."""
    sheets = []
    for start in range(0, len(video), 24):
        panel = Image.new('RGB', (6 * 192, 4 * 152), 'white')
        draw = ImageDraw.Draw(panel)
        indices = list(range(start, min(start + 24, len(video))))
        for index in indices:
            image = Image.fromarray(video[index])
            image.thumbnail((192, 128))
            x, y = ((index - start) % 6) * 192, ((index - start) // 6) * 152
            panel.paste(image, (x, y + 20))
            draw.text((x + 3, y + 2), f'frame {index}', fill='black')
        name = f'rgb_{start:03d}.png'
        panel.save(folder / name)
        sheets.append({'path': name, 'sha256': sha(folder / name), 'frame_indices': indices})
    save(folder / 'visual_judge_payload.json', {'opaque_case_id': opaque_id,
         'frames': len(video), 'sheets': sheets, 'status': 'pending',
         'instruction': 'Report PASS/FAIL/ABSTAIN using visible evidence and frame ranges; abstain if synchronized release, camera or contact cannot be established. Do not infer unobserved motion.'})


def load_video(path):
    with np.load(path) as source:
        key = 'video' if 'video' in source.files else 'frames'
        return source[key]


def process(case, folder, project, modules, models=None):
    files = case['files']
    source = verify_file(files['input'])
    domain = case['domain']
    row = {'id': case['id'], 'domain': domain, 'status': 'ok', 'stages': {
        'input': {'status': 'executed', 'path': str(source), 'sha256': files['input']['sha256']}}}
    stage = row['stages']
    arrays = {}
    if domain == 'control':
        setup = json.loads(verify_file(files['setup']).read_text())
        measurement = modules['gravity'].measure_video(source, setup)
        row['frames'] = measurement.get('decoded_frames')
        centers = measurement.get('tracks', [[], []])
        arrays['co_xy'] = np.array([[p['center'] if p else [np.nan, np.nan] for p in track] for track in centers]).transpose(1, 0, 2)
        arrays['co_visible'] = np.isfinite(arrays['co_xy']).all(axis=-1)
        stage['tracking'] = {'status': 'executed', 'method': 'frozen_colored_object_centroid'}
        stage['observation'] = {'status': 'executed', 'reason': measurement.get('reason')}
        stage['depth'] = stage['3d'] = {'status': 'not_applicable', 'reason': 'dimensionless_planar_relation'}
        row['relation'] = relation_decision(measurement, 'score' in measurement)
        row['measurement'] = measurement
        if 'judgment' in files:
            judgment = json.loads(verify_file(files['judgment']).read_text())
            row['visual_judge'] = {'status': 'reused', 'verdict': judgment['verdict'],
                                   'independent_session_rerun': False}
        else:
            row['visual_judge'] = {'status': 'pending'}
        stage['judge'] = {'status': 'executed', 'channel': 'numeric_relation', 'visual_channel': row['visual_judge']['status']}
        row['evaluation'] = {}
        if 'truth' in files:
            # Evaluation-only: measurement above has already finished.
            labels = json.loads(verify_file(files['truth']).read_text())
            target = next(t for t in labels if t['case_id'] == case['reference_key'])
            row['reference'] = {key: target[key] for key in ['scene_seed', 'condition', 'expected']}
            row['evaluation'] = {'relation': {'correct': int(row['relation']['verdict'] == target['expected'])}}
    else:
        with np.load(source) as inp:
            if domain in ['davis', 'pstudio']:
                query = inp['queries_tyx']
                k = inp['intrinsics'] if domain == 'pstudio' else None
                frame_count = case['frames']
            else:
                setup = case['setup']
                offset = 32 * np.array([[0, 0], [-.4, 0], [.4, 0], [0, -.4], [0, .4]], np.float32)
                xy0 = np.concatenate([np.array(c, np.float32) + offset for c in setup['initial_centers']])
                query = np.concatenate([np.zeros((10, 1), np.float32), xy0[:, ::-1]], axis=-1)
                k = np.array([[768, 0, 383.5], [0, 768, 255.5], [0, 0, 1]], np.float32)
                frame_count = 97
        row['frames'] = frame_count
        arrays['queries_tyx'] = query
        if domain == 'davis':
            pred = cached(case, 'prediction')
            arrays.update(co_xy=pred['xy'], co_visible=pred['visible'])
            stage['tracking'] = {'status': 'reused', 'method': 'native_cotracker3_querywise'}
            stage['observation'] = {'status': 'executed', 'policy': 'native_visibility', 'independent_identity_validated': False}
            stage['depth'] = stage['3d'] = {'status': 'not_applicable', 'reason': 'no_camera_or_metric_depth_labels'}
        else:
            if domain == 'i2v' and models is not None:
                video = load_video(source)
                pred, inference = models.run(video, query, k)
                np.savez_compressed(folder / 'fresh_predictions.npz', **pred)
                save(folder / 'fresh_inference.json', {**inference, 'status': 'ok',
                     'input_sha256': files['input']['sha256'], 'prediction_sha256': sha(folder / 'fresh_predictions.npz')})
                mode = 'executed'
            elif domain == 'pstudio':
                pred = cached(case, 'prediction')
                mode = 'reused'
            else:
                co, tap, depth = (cached(case, name) for name in ['co', 'tap', 'depth'])
                if not np.array_equal(co['queries'][:, [0, 2, 1]], query):
                    raise ValueError('Cached queries do not match inference specification')
                pred = {'co_xy': co['xy'], 'co_visible': co['official_visible'],
                        'tap_xy': tap['xy'], 'tap_xyz': tap['xyz_assumed_camera'],
                        'tap_visible': tap['official_visible'], 'depth': depth['depth']}
                mode = 'reused'
            depth = pred['depth']
            if depth.ndim != 3 or len(depth) != frame_count or not np.isfinite(depth).all() or not (depth > 0).all():
                raise ValueError('Depth cache shape/range mismatch')
            co_xyz, valid = lift_points(pred['co_xy'], depth, k)
            arrays.update({key: pred[key] for key in ['co_xy', 'co_visible', 'tap_xy', 'tap_visible', 'tap_xyz']})
            arrays.update(co_xyz=co_xyz, co_valid=valid, intrinsics=k)
            stage['tracking'] = {'status': mode, 'methods': ['cotracker3', 'tapip3d']}
            stage['depth'] = {'status': mode, 'method': 'DA-V2_metric_hypersim_vits', 'shape': list(depth.shape), 'metric_accuracy_validated': False}
            stage['3d'] = {'status': 'executed', 'methods': ['co_plus_depth', 'native_tapip3d'],
                          'camera': 'supplied_static' if domain == 'pstudio' else 'assumed_static_fx_fy_width',
                          'co_outside_or_invalid_points': int((~valid).sum())}
        for prefix in ['co', 'tap']:
            if prefix + '_xy' not in arrays:
                continue
            xy, visible = arrays[prefix + '_xy'], arrays[prefix + '_visible']
            if xy.shape != (frame_count, len(query), 2) or visible.shape != xy.shape[:2]:
                raise ValueError('Tracking dimensions mismatch')
            if visible.dtype != bool or not np.isfinite(xy).all():
                raise ValueError('Invalid native tracking coordinates/visibility')
        if domain == 'i2v':
            video = load_video(source)
            masks = np.load(verify_file(files['masks']))
            obs = modules['i2v'].assess_frames(video, setup, [masks['left'], masks['right']])
            raw_video = verify_file(files['lossless_video'])
            cap = cv2.VideoCapture(str(raw_video))
            for expected in video:
                ok, decoded = cap.read()
                if not ok or not np.array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), expected):
                    raise ValueError('Relation video differs from inference RGB')
            extra, _ = cap.read()
            cap.release()
            if extra:
                raise ValueError('Unexpected extra relation video frame')
            measurement = modules['gravity'].measure_video(raw_video, {**setup, 'frames': 97, 'fps': 24})
            contact, reason = modules['i2v'].contact_wrapper(measurement, setup)
            admissible = all(obs[key] for key in ['initial_objects_passed', 'initial_setup_passed', 'coverage_passed',
                'individual_fiducial_camera_passed', 'floor_observability_passed']) and contact
            row.update(observation=obs, measurement=measurement,
                       relation=relation_decision(measurement, admissible))
            row['relation'].update(contact_reason=reason, raw_diagnostic_score=measurement.get('score'))
            stage['observation'] = {'status': 'executed', 'method': 'frozen_RGB_initial_coverage_camera_floor_gate', 'passed': bool(admissible)}
            row['visual_judge'] = {'status': 'pending', 'human_labels_available': False}
            stage['judge'] = {'status': 'executed', 'channel': 'numeric_relation', 'visual_channel': 'pending'}
            blind = folder / 'blind_visual_input'
            blind.mkdir()
            rgb_sheet(video, blind, opaque_id(case['id']))
            row['evaluation'] = {}
        else:
            # Only this evaluator opens reference coordinates/visibility.
            truth = np.load(verify_file(files['truth']))
            if domain == 'davis':
                metrics = modules['official'].compute_tapvid_metrics(query[None], truth['occluded'][None],
                    truth['target_points'][None], (~arrays['co_visible'].T)[None], arrays['co_xy'].transpose(1, 0, 2)[None], 'first')
                row['evaluation'] = {'co': {key: float(metrics[key][0]) for key in ['average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy']}}
            else:
                positive = truth['visible'] & (np.arange(frame_count)[:, None] > query[None, :, 0])
                row['evaluation'] = {}
                for prefix in ['co', 'tap']:
                    xyz = arrays[prefix + '_xyz']
                    row['evaluation'][prefix] = {'raw_epe3d_m': float(np.linalg.norm(xyz - truth['tracks_xyz'], axis=-1)[positive].mean()),
                        'epe2d_px': float(np.linalg.norm(arrays[prefix + '_xy'] - truth['tracks_xy'], axis=-1)[positive].mean()),
                        'visible_point_times': int(positive.sum())}
                stage['observation'] = {'status': 'executed', 'policy': 'native_visibility', 'reference_type': 'geometry_derived'}
            row['relation'] = {'verdict': 'NOT_APPLICABLE', 'score': None, 'reason': 'no_controlled_two_object_physics_setup'}
            stage['judge'] = {'status': 'not_applicable', 'reason': 'measurement_benchmark_only'}
    stage['relation'] = {'status': 'executed' if domain in ['control', 'i2v'] else 'not_applicable', 'verdict': row['relation']['verdict']}
    stage['evaluation'] = {'status': 'executed' if row['evaluation'] else 'not_applicable',
                           'future_truth_used_only_here': True}
    np.savez_compressed(folder / 'tracks.npz', **arrays)
    stage['tracking']['artifact'] = 'tracks.npz'
    stage['tracking']['artifact_sha256'] = sha(folder / 'tracks.npz')
    return row


def run(manifest, output, fresh_i2v=False, gpu=1, visual_opinions=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    ids = [c['id'] for c in manifest['cases']]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in ids):
        raise ValueError('Case IDs must be unique, safe directory names')
    opinions = index_opinions(visual_opinions or [], {opaque_id(c['id']) for c in manifest['cases'] if c['domain'] == 'i2v'})
    output.mkdir(parents=True)
    save(output / 'manifest.json', manifest)
    project = Path(manifest.get('project', Path(__file__).resolve().parents[3]))
    modules = {name: module(verify_file(entry), 'integrated_' + name) for name, entry in manifest.get('sources', {}).items()}
    if 'judge_calibration' in manifest:
        calibration = manifest['judge_calibration']
        script = verify_file(calibration['script'])
        receipt = verify_file(calibration['receipts'])
        command = [sys.executable, str(script), '--study', calibration['study'], '--receipts', str(receipt), '--output', str(output / 'judge_calibration.json')]
        result = subprocess.run(command, capture_output=True, text=True)
        save(output / 'judge_calibration_run.json', {'command': command, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError('Control judge calibration failed; see run receipt')
        calibration_valid = json.loads((output / 'judge_calibration.json').read_text())['judge_passed']
    else:
        calibration_valid = None
    models = None
    model_error = None
    if fresh_i2v:
        from .backends import NativeModels
        try:
            models = NativeModels(project, gpu)
        except Exception as error:
            model_error = repr(error)
            save(output / 'model_initialization_error.json', {'status': 'error', 'error': model_error})
    rows = []
    with (output / 'events.jsonl').open('x') as log:
        for case in manifest['cases']:
            folder = output / case['id']
            folder.mkdir()
            print(json.dumps({'case': case['id'], 'event': 'start'}), flush=True)
            try:
                if fresh_i2v and case['domain'] == 'i2v' and model_error is not None:
                    raise RuntimeError('Fresh inference initialization failed: ' + model_error)
                row = process(case, folder, project, modules, models)
                opinion = opinions.get(opaque_id(case['id']))
                if case['domain'] == 'i2v' and opinion is not None:
                    row['visual_judge'] = validate_opinion(opinion, folder / 'blind_visual_input/visual_judge_payload.json')
                    row['stages']['judge']['visual_channel'] = 'diagnostic_import'
                    save(folder / 'imported_visual_opinion.json', opinion)
                if row['domain'] in ['control', 'i2v']:
                    row['judge_calibration_valid'] = calibration_valid
                    if calibration_valid is not True:
                        row['relation'] = {'verdict': 'ABSTAIN', 'score': None, 'reason': 'judge_calibration_invalid_or_missing'}
                        row['stages']['relation']['verdict'] = 'ABSTAIN'
                        if 'reference' in row:
                            row['evaluation']['relation']['correct'] = int(row['relation']['verdict'] == row['reference']['expected'])
                save(folder / 'result.json', row)
            except Exception as error:
                row = {'id': case['id'], 'domain': case['domain'], 'status': 'error', 'error': repr(error)}
                save(folder / 'error.json', row)
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps({'case': case['id'], 'event': 'done', 'status': row['status']}), flush=True)
    summary = aggregate_rows(rows)
    summary.update(judge_calibration_valid=calibration_valid, fresh_i2v_requested=fresh_i2v,
                   framework_execution_complete=True, physical_measurement_validated_for_i2v=False)
    save(output / 'summary.json', summary)
    save(output / 'cases.json', rows)
    save(output / 'run_receipt.json', {'utc': datetime.now(timezone.utc).isoformat(),
         'manifest_sha256': sha(output / 'manifest.json'), 'python': sys.version, 'command': sys.argv,
         'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
         'visual_opinions': list(opinions.values()),
         'threshold_optimization_performed': False, 'new_generation_performed': False})
    lines = ['# Integrated framework run', '', f'Cases: {len(rows)}; errors: {summary["errors"]}.',
             f'Control visual-judge calibration: {calibration_valid}.', '',
             '|Domain|Cases|Errors|Relation decisions|', '|---|---:|---:|---|']
    for name, data in summary['by_domain'].items():
        lines.append(f'|{name}|{data["cases"]}|{data["errors"]}|{data["verdicts"]}|')
    lines += ['', 'Per-case stage records: cases.json and individual result.json files.',
              'Fresh and reused inference are explicitly distinguished. Missing visual judgments remain pending.',
              'Framework execution is separate from physical validity. See the Chinese study report for limitations.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary
