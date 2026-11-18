"""Read-only checkpoint/video screening; unlabeled outputs are not accuracy."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'annotation_frames').mkdir()
    (out / 'predicted_masks').mkdir()
    sys.path.insert(0, str(source / 'benchmark'))
    from training_pilot import TinyUNet, make_dataset, evaluate, mask_metrics

    torch.set_num_threads(8)
    checkpoint = source / 'benchmark/results/training_pilot/best_model.pt'
    old = json.loads((checkpoint.parent / 'metrics.json').read_text())
    assert sha(checkpoint) == old['model_sha256'], 'Checkpoint changed'
    assert sha(source / 'benchmark/training_pilot.py') == old['source_sha256']
    model = TinyUNet().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    x, y, _ = make_dataset(256, 30001)
    assert hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() == old['dataset_hashes']['test']
    with torch.no_grad():
        learned_test = evaluate(model, x, y)
    color = ((x[:, 0] > 80/255) & (x[:, 0] > 1.4*x[:, 1]) & (x[:, 0] > 1.4*x[:, 2])).float()[:, None]
    synthetic = {'learned': learned_test, 'color': mask_metrics(color, y),
                 'scope': 're-evaluation of existing synthetic test set; no new independent replication'}
    dump(out / 'synthetic_recheck.json', synthetic)
    rows_path = source / 'benchmark/results/raw/part_a.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == len({(r['model'], r['prompt_id'], r['seed']) for r in rows}) == 24
    assert {r['model'] for r in rows} == {'wan2.1_1.3b', 'cogvideox_5b'}
    assert all(sum(r['model'] == m for r in rows) == 12 for m in {r['model'] for r in rows})
    frames_out, clips_out, annotations, key = [], [], [], []
    rng = np.random.default_rng(20261109)
    order = rng.permutation(len(rows))
    for blind_index, row_index in enumerate(order):
        row = rows[int(row_index)]
        blind = f'clip_{blind_index:03d}'
        video = Path(row['video_path'])
        assert sha(video) == row['video_sha256']
        meta = json.loads(video.with_suffix('.json').read_text())
        assert meta['video_sha256'] == row['video_sha256']
        with av.open(str(video)) as container:
            fps = float(container.streams.video[0].average_rate)
            original = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        assert len(original) == meta['frames'] and fps == meta['fps'] == row['fps']
        # Screening uses the training input size; this distorts aspect ratio.
        # It is deliberately not a replacement for the full-resolution tracker.
        small = np.stack([cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA) for frame in original])
        tensor = torch.from_numpy(small.transpose(0, 3, 1, 2).copy()).float()/255
        with torch.no_grad():
            learned = torch.cat([model(batch).sigmoid()[:, 0] for batch in tensor.split(32)]).numpy() > .5
        red, green, blue = small.astype(np.float32).transpose(3, 0, 1, 2)
        classical = (red > 80) & (red > 1.4*green) & (red > 1.4*blue)
        selected = set(np.linspace(0, len(original)-1, 5).round().astype(int).tolist())
        frame_rows = []
        for i, (pm, cm) in enumerate(zip(learned, classical)):
            intersection, union = int((pm & cm).sum()), int((pm | cm).sum())
            py, px = np.nonzero(pm)
            cy, cx = np.nonzero(cm)
            learned_present, color_present = bool(pm.any()), bool(cm.any())
            fr = {'blind_clip': blind, 'frame': i, 'learned_nonempty': learned_present,
                  'color_nonempty': color_present, 'mask_iou_between_methods': intersection/union if union else None,
                  'centroid_disagreement_96px': float(np.hypot(px.mean()-cx.mean(), py.mean()-cy.mean())) if learned_present and color_present else None,
                  'learned_area_96px': int(pm.sum()), 'color_area_96px': int(cm.sum())}
            frames_out.append(fr)
            frame_rows.append(fr)
            if i in selected:
                name = f'{blind}_frame_{i:03d}.png'
                cv2.imwrite(str(out/'annotation_frames'/name), cv2.cvtColor(original[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out/'predicted_masks'/f'{blind}_{i:03d}_learned.png'), pm.astype(np.uint8)*255)
                cv2.imwrite(str(out/'predicted_masks'/f'{blind}_{i:03d}_color.png'), cm.astype(np.uint8)*255)
                annotations.append({'frame_file': name, 'target_state': '', 'cx_original_px': '',
                                    'cy_original_px': '', 'mask_file': '', 'annotator': '', 'notes': ''})
        valid_iou = [f['mask_iou_between_methods'] for f in frame_rows if f['mask_iou_between_methods'] is not None]
        clips_out.append({'blind_clip': blind, 'model': row['model'], 'prompt_id': row['prompt_id'], 'seed': row['seed'],
                          'frames': len(original), 'learned_nonempty_fraction': float(learned.any((1,2)).mean()),
                          'color_nonempty_fraction': float(classical.any((1,2)).mean()),
                          'mean_method_iou_nonempty_union': float(np.mean(valid_iou)) if valid_iou else None,
                          'frozen_track_status': row['track_status'], 'frozen_fit_status': row['fit_status']})
        key.append({'blind_clip': blind, 'video_path': str(video), 'sha256': row['video_sha256'],
                    'model': row['model'], 'prompt_id': row['prompt_id'], 'seed': row['seed']})
        print(f'{blind}: {len(original)} frames screened', flush=True)
    with (out / 'annotation_template.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)
    dump(out/'frames.json', frames_out)
    dump(out/'clips.json', clips_out)
    dump(out/'restricted_blinding_key.json', key)
    summary = {}
    for name in sorted({r['model'] for r in clips_out}):
        group = [r for r in clips_out if r['model'] == name]
        summary[name] = {'n_clips': len(group), 'n_frames': sum(r['frames'] for r in group),
                         'mean_clip_learned_nonempty_fraction': float(np.mean([r['learned_nonempty_fraction'] for r in group])),
                         'mean_clip_color_nonempty_fraction': float(np.mean([r['color_nonempty_fraction'] for r in group])),
                         'mean_clip_method_iou': float(np.mean([r['mean_method_iou_nonempty_union'] for r in group if r['mean_method_iou_nonempty_union'] is not None])),
                         'frozen_track_ok': sum(r['frozen_track_status']=='ok' for r in group),
                         'frozen_fit_ok': sum(r['frozen_fit_status']=='ok' for r in group)}
    dump(out/'summary.json', {'models': summary, 'annotation_frames': len(annotations),
                             'scope': 'unlabeled screening; nonempty masks and agreement are not accuracy or valid tracking'})
    dump(out/'provenance.json', {'source_root': str(source), 'source_script_sha256': sha(__file__),
                               'training_source_sha256': sha(source/'benchmark/training_pilot.py'),
                               'checkpoint_sha256': sha(checkpoint), 'input_jsonl_sha256': sha(rows_path),
                               'seed_blinding': 20261109, 'device': 'cpu', 'threshold': .5,
                               'resize': '96x96 INTER_AREA, direct resize (aspect distorted)',
                               'python': sys.version,
                               'versions': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'opencv-python-headless', 'av']}})


if __name__ == '__main__':
    main()
