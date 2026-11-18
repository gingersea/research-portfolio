"""Fixed native inference, separate from all reference-label evaluators."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .contracts import lift_points, sha


class NativeModels:
    def __init__(self, project, gpu=1):
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        torch.manual_seed(20261115)
        torch.cuda.set_device(gpu)
        self.device = f'cuda:{gpu}'
        research = Path(project) / 'research'
        self.assets = {name: json.loads((research / path).read_text()) for name, path in [
            ('co', 'tracking_validation_20261115/assets.json'),
            ('tap', 'observation_gate_20261115/tapip_assets.json'),
            ('depth', 'identity_recovery_20261115/depth_assets.json')]}
        tap_sources = json.loads((research / 'observation_gate_20261115/tapip_source.json').read_text())['files']
        for name, assets in self.assets.items():
            if sha(assets['checkpoint']) != assets['checkpoint_sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            sources = assets.get('source_files', assets.get('files', tap_sources))
            for row in sources:
                if sha(Path(assets['source_directory']) / row['file']) != row['sha256']:
                    raise ValueError('Native source checksum mismatch')
        co = self.assets['co']
        sys.path.insert(0, co['source_directory'])
        from cotracker.predictor import CoTrackerPredictor
        self.co = CoTrackerPredictor(checkpoint=co['checkpoint'], offline=True, v2=False, window_len=60).to(self.device).eval()
        source = Path(self.assets['tap']['source_directory'])
        sys.path[:0] = [str(source), str(source / 'third_party/pointops2')]
        from utils.inference_utils import load_model, inference
        self.inference = inference
        original = torch.hub.load

        def local(repo, name, *args, **kwargs):
            if repo != 'facebookresearch/co-tracker' or name != 'cotracker3_offline' or args or kwargs:
                raise ValueError('Unexpected model dependency')
            return CoTrackerPredictor(checkpoint=co['checkpoint'], offline=True, v2=False, window_len=60)

        torch.hub.load = local
        try:
            self.tap = load_model(self.assets['tap']['checkpoint']).to(self.device).eval()
        finally:
            torch.hub.load = original
        state = torch.load(self.assets['tap']['checkpoint'], map_location='cpu', weights_only=False)['weight']
        if not all(torch.equal(self.tap.state_dict()[key].cpu(), value) for key, value in state.items()):
            raise ValueError('TAPIP final weights differ from checkpoint')
        self.exact_tensors = len(state)
        del state
        source = Path(self.assets['depth']['source_directory'])
        sys.path.insert(0, str(source / 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.dinov2_layers import attention
        attention.XFORMERS_AVAILABLE = False
        self.depth = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
        self.depth.load_state_dict(torch.load(self.assets['depth']['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        self.depth = self.depth.to(self.device).eval()

    def run(self, video, queries_tyx, k):
        torch = self.torch
        started = time.monotonic()
        t, h, w = video.shape[:3]
        query = queries_tyx[:, [0, 2, 1]].copy()
        if np.any(query[:, 0] != 0):
            raise ValueError('Fresh I2V adapter expects initial-frame queries')
        with torch.inference_mode():
            frames = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(self.device)
            xy, visible = self.co(video=frames[None], queries=torch.from_numpy(query)[None].to(self.device), backward_tracking=False)
            co_xy, co_visible = xy[0].cpu().numpy(), visible[0].cpu().numpy()
            depth = np.stack([self.depth.infer_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), input_size=518) for frame in video]).astype(np.float32)
            if depth.shape != (t, h, w) or not np.isfinite(depth).all() or not (depth > 0).all():
                raise ValueError('Invalid native depth output')
            initial, valid = lift_points(query[None, :, 1:], depth[:1], k)
            if not valid.all():
                raise ValueError('Invalid query depth')
            query3d = np.concatenate([query[:, :1], initial[0]], axis=-1).astype(np.float32)
            xyz, vis = self.inference(model=self.tap, video=frames / 255.,
                depths=torch.from_numpy(depth).to(self.device),
                intrinsics=torch.from_numpy(np.repeat(k[None], t, axis=0)).to(self.device),
                extrinsics=torch.eye(4, device=self.device)[None].repeat(t, 1, 1),
                query_point=torch.from_numpy(query3d).to(self.device), num_iters=6,
                grid_size=6, vis_threshold=.9, bidrectional=True)
            tap_xyz, tap_visible = xyz.cpu().numpy(), vis.cpu().numpy()
        pixels = tap_xyz @ k.T
        co_xyz, co_valid = lift_points(co_xy, depth, k)
        return {'depth': depth, 'co_xy': co_xy, 'co_visible': co_visible,
                'co_xyz': co_xyz, 'co_valid': co_valid, 'tap_xyz': tap_xyz,
                'tap_xy': pixels[..., :2] / pixels[..., 2:], 'tap_visible': tap_visible}, {
            'mode': 'fresh', 'elapsed_seconds': time.monotonic() - started,
            'checkpoint_sha256': {name: a['checkpoint_sha256'] for name, a in self.assets.items()},
            'future_truth_read': False, 'iterations': 6, 'support_grid': 6,
            'depth_input_size': 518, 'depth_max': 20, 'dtype': 'float32',
            'tap_final_state_tensors_equal_checkpoint': self.exact_tensors,
            'attention_backend': 'author_native_pytorch', 'i2v_queries_inferred_jointly': True}
