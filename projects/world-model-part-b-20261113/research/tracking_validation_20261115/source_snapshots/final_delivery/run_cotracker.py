"""Official native predictor with a read-only raw-output hook; RGB/query inputs only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'heldout'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-name', default='primary')
    parser.add_argument('--only', default=None)
    parser.add_argument('--data-manifest', default='data_manifest.json')
    args = parser.parse_args()
    assert args.run_name.replace('_', '').isalnum()
    assets = json.loads((ROOT/'assets.json').read_text())
    checkpoint = Path(assets['checkpoint'])
    assert sha(checkpoint) == assets['checkpoint_sha256']
    source_dir = Path(assets['source_directory'])
    assert all(sha(source_dir/item['file']) == item['sha256'] for item in assets['source_files'])
    sys.path.insert(0, str(source_dir))
    from cotracker.predictor import CoTrackerPredictor
    torch.set_num_threads(4)
    device = f'cuda:{args.gpu}'
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False, window_len=60).to(device).eval()
    captured = []

    native_forward = model.model.forward

    def observe_forward(*inputs, **keywords):
        outputs = native_forward(*inputs, **keywords)
        captured.append((outputs[1].detach().cpu().numpy().copy(), outputs[2].detach().cpu().numpy().copy()))
        return outputs

    # Native predictor invokes .forward directly, bypassing Module.__call__ hooks.
    model.model.forward = observe_forward
    manifest = json.loads((ROOT/args.data_manifest).read_text())
    assert manifest['protocol_sha256'] == sha(ROOT/'PROTOCOL.md')
    output = ROOT/'predictions'/args.run_name/args.split
    output.mkdir(parents=True, exist_ok=True)
    for case in manifest['cases']:
        if case['split'] != args.split or args.only and case['case'] != args.only:
            continue
        record_path = output/f'{case["case"]}.json'
        if record_path.exists():
            raise FileExistsError('No quality retry or replacement of existing attempts')
        # No future trajectory, depth, instance map or visibility labels are read.
        input_path = ROOT/case['input']
        assert sha(input_path) == case['input_sha256']
        inputs = np.load(input_path, allow_pickle=False)
        rgb = inputs['video']
        query = inputs['queries']
        video = torch.from_numpy(rgb).permute(0,3,1,2).float()[None].to(device)
        queries = torch.from_numpy(query)[None].to(device)
        captured.clear()
        started = time.monotonic()
        print(json.dumps({'event': 'start', 'case': case['case']}), flush=True)
        try:
            with torch.inference_mode():
                xy, official_visible = model(video=video, queries=queries, backward_tracking=False)
            assert len(captured) == 1
            raw_vis, raw_conf = [array[0, :, :len(query)] for array in captured[0]]
            predicted = xy[0].cpu().numpy()
            visible = official_visible[0].cpu().numpy()
            assert predicted.shape == (33, 10, 2) and np.isfinite(predicted).all()
            assert raw_vis.shape == raw_conf.shape == visible.shape == (33, 10)
            prediction_path = output/f'{case["case"]}.npz'
            np.savez_compressed(prediction_path, xy=predicted, official_visible=visible,
                                visibility=raw_vis, confidence=raw_conf,
                                conservative_visible=(raw_vis*raw_conf > .9))
            record = {'status': 'ok', 'case': case['case'], 'input_sha256': case['input_sha256'],
                      'prediction_sha256': sha(prediction_path), 'checkpoint_sha256': assets['checkpoint_sha256'],
                      'data_manifest_sha256': sha(ROOT/args.data_manifest),
                      'source_revision': assets['source_revision'], 'runner_sha256': sha(Path(__file__)),
                      'protocol_sha256': sha(ROOT/'PROTOCOL.md'), 'future_truth_read': False,
                      'read_only_hook': True, 'observer_kind': 'forward_method_return_observer', 'native_query_frame_forced': True,
                      'model_resolution': list(model.interp_shape), 'support_grid': 6,
                      'iterations': 6, 'dtype': 'float32', 'frames': 33, 'queries': 10,
                      'elapsed_seconds': time.monotonic()-started,
                      'maximum_gpu_allocated_bytes': torch.cuda.max_memory_allocated(args.gpu)}
        except Exception as error:
            record = {'status': 'api_failure', 'case': case['case'], 'exception': repr(error),
                      'elapsed_seconds': time.monotonic()-started, 'runner_sha256': sha(Path(__file__))}
            record_path.write_text(json.dumps(record, indent=2)+'\n')
            raise
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'event': 'complete', **record}), flush=True)
    model.model.forward = native_forward


if __name__ == '__main__':
    main()
