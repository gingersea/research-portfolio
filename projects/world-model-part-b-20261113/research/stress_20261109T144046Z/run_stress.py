"""Frozen segmentation stress test with procedurally known synthetic masks.

This is a measurement study, not a new real-video accuracy benchmark.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def metrics(pred, truth):
    intersection = (pred & truth).sum()
    union = (pred | truth).sum()
    py, px = np.nonzero(pred)
    ty, tx = np.nonzero(truth)
    assert len(tx), 'Every positive synthetic scene must contain a target'
    return {'iou': float(intersection/union),
            'centroid_error_px': float(np.hypot(px.mean()-tx.mean(), py.mean()-ty.mean())) if len(px) else float(np.hypot(*truth.shape)),
            'nonempty': float(bool(len(px)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo/'benchmark'))
    from training_pilot import TinyUNet, make_dataset
    train_record = json.loads(args.checkpoint.with_name('metrics.json').read_text())
    assert sha(args.checkpoint) == train_record['model_sha256']
    assert sha(repo/'benchmark/training_pilot.py') == train_record['source_sha256']
    torch.set_num_threads(8)
    model = TinyUNet().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    x, y, _ = make_dataset(256, seed=40001)
    images = (x.numpy().transpose(0,2,3,1)*255).round().astype(np.uint8)
    masks = y.numpy()[:,0] > .5
    assert metrics(masks[0], masks[0])['iou'] == 1
    assert metrics(np.zeros_like(masks[0]), masks[0])['centroid_error_px'] == np.hypot(96,96)
    rows = []
    for patch_size in [24,48,96]:
        original_images, original_masks = [], []
        for im, mask in zip(images, masks):
            canvas = np.full((112,192,3),255,dtype=np.uint8)
            truth = np.zeros((112,192),dtype=np.uint8)
            cy, cx = (112-patch_size)//2, (192-patch_size)//2
            canvas[cy:cy+patch_size,cx:cx+patch_size] = cv2.resize(im,(patch_size,patch_size),interpolation=cv2.INTER_AREA)
            truth[cy:cy+patch_size,cx:cx+patch_size] = cv2.resize(mask.astype(np.uint8),(patch_size,patch_size),interpolation=cv2.INTER_NEAREST)
            original_images.append(canvas)
            original_masks.append(truth.astype(bool))
        image_dir = out/'scenes'/f'patch_{patch_size}'
        image_dir.mkdir(parents=True)
        for i,(im,truth) in enumerate(zip(original_images,original_masks)):
            cv2.imwrite(str(image_dir/f'{i:03d}_rgb.png'), cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(image_dir/f'{i:03d}_truth.png'), truth.astype(np.uint8)*255)
        for strategy in ['native','warp96','letterbox96']:
            batch = []
            for im in original_images:
                if strategy=='native':
                    transformed = im
                elif strategy=='warp96':
                    transformed = cv2.resize(im,(96,96),interpolation=cv2.INTER_AREA)
                else:
                    transformed = np.full((96,96,3),255,dtype=np.uint8)
                    transformed[20:76] = cv2.resize(im,(96,56),interpolation=cv2.INTER_AREA)
                batch.append(transformed)
            data = np.stack(batch)
            tensor = torch.from_numpy(data.transpose(0,3,1,2).copy()).float()/255
            with torch.no_grad():
                predicted = torch.cat([model(b).sigmoid()[:,0] for b in tensor.split(32)]).numpy() > .5
            red,green,blue = data.astype(np.float32).transpose(3,0,1,2)
            baseline = (red>80)&(red>1.4*green)&(red>1.4*blue)
            for method, method_masks in [('learned',predicted),('color',baseline)]:
                dest=out/'predictions'/f'patch_{patch_size}'/strategy/method
                dest.mkdir(parents=True)
                for i,(pm,truth) in enumerate(zip(method_masks,original_masks)):
                    if strategy=='native':
                        restored = pm
                    else:
                        if strategy=='letterbox96': pm=pm[20:76]
                        restored=cv2.resize(pm.astype(np.uint8),(192,112),interpolation=cv2.INTER_NEAREST).astype(bool)
                    cv2.imwrite(str(dest/f'{i:03d}.png'), restored.astype(np.uint8)*255)
                    rows.append({'scene_id':i,'patch_size':patch_size,'strategy':strategy,'method':method,
                                 'true_diameter_area_equivalent_px':float(2*np.sqrt(truth.sum()/np.pi)),
                                 **metrics(restored,truth)})
            print(f'patch {patch_size}, {strategy}: 256 paired scenes',flush=True)
    write_json(out/'per_scene.json',rows)
    rng=np.random.default_rng(20261110)
    indices=rng.integers(0,256,size=(2000,256))
    summary=[]
    for size in [24,48,96]:
        for strategy in ['native','warp96','letterbox96']:
            groups={m:[r for r in rows if r['patch_size']==size and r['strategy']==strategy and r['method']==m] for m in ['learned','color']}
            for method, selected in groups.items():
                record={'patch_size':size,'strategy':strategy,'method':method,'n_scenes':len(selected)}
                for metric in ['iou','centroid_error_px','nonempty']:
                    values=np.array([r[metric] for r in selected])
                    record[metric]={'mean':float(values.mean()),'ci95':np.quantile(values[indices].mean(1),[.025,.975]).tolist()}
                summary.append(record)
    paired=[]
    for size in [24,48,96]:
        for method in ['learned','color']:
            native=np.array([r['iou'] for r in rows if r['patch_size']==size and r['strategy']=='native' and r['method']==method])
            for strategy in ['warp96','letterbox96']:
                values=np.array([r['iou'] for r in rows if r['patch_size']==size and r['strategy']==strategy and r['method']==method])-native
                paired.append({'patch_size':size,'method':method,'contrast':strategy+' minus native','n_paired_scenes':256,
                               'mean_iou_difference':float(values.mean()),'ci95':np.quantile(values[indices].mean(1),[.025,.975]).tolist()})
    write_json(out/'summary.json',summary)
    write_json(out/'paired_contrasts.json',paired)
    write_json(out/'provenance.json',{'script_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(args.checkpoint),
                'training_source_sha256':sha(repo/'benchmark/training_pilot.py'),
                'dataset_tensor_sha256':hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest(),
                'render_seed':40001,'bootstrap_seed':20261110,'bootstrap_resamples':2000,
                'confidence_level':.95,'n_scenes_per_condition':256,'independent_seed_stream_from_training':True,
                'scene_canvas_hw':[112,192],'patch_sizes':[24,48,96],
                'native_model_input_hw':[112,192],'warp_input_hw':[96,96],'letterbox_content_hw':[56,96],
                'letterbox_top_bottom_padding':20,'mask_threshold':.5,'prediction_inverse':'nearest neighbor',
                'empty_prediction_centroid_penalty_px':float(np.hypot(112,192)),
                'python':sys.version,'torch_runtime':str(torch.__version__),'device':'cpu',
                'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','opencv-python-headless']},
                'scope':'synthetic positive scenes only; no human labels, training or real-video scoring'})
    assert len(rows)==256*3*3*2
    print('STRESS_COMPLETE: 4608 per-scene measurements; labels are procedural synthetic truth')


if __name__=='__main__':
    main()
