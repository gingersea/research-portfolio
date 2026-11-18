"""Exact RGB/depth/identity controls; predictions never enter label generation."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
H, W, T, FOCAL, RADIUS = 256, 384, 33, 320., .22
CONDITIONS = ['clean', 'textureless', 'hue_change', 'same_hue', 'occlusion',
              'disappearance', 'camera_motion', 'crossing', 'identity_replacement', 'ambiguous_twins']
OFFSETS = RADIUS*np.array([[0, 0], [-.4, 0], [.4, 0], [0, -.4], [0, .4]])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(seed, condition):
    rng = np.random.default_rng(seed)
    K = np.array([[FOCAL, 0, (W-1)/2], [0, FOCAL, (H-1)/2], [0, 0, 1]], dtype=np.float32)
    base = np.array([[-.78, -.12, 3.8], [.78, .12, 4.4]])
    base[:, :2] += rng.uniform(-.06, .06, (2, 2))
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    frames, depths, positions, projections, visibility, cameras, identifiable = [], [], [], [], [], [], []
    maps = []
    for t in range(T):
        tau = t/(T-1)
        center = base.copy()
        center[:, 0] += .16*np.sin(2*np.pi*tau+np.array([0, .7]))
        center[:, 1] += .25*np.sin(np.pi*tau)
        camera = np.zeros(3)
        present = [True, True]
        if condition == 'camera_motion':
            center = base.copy()
            camera = np.array([.4*np.sin(2*np.pi*tau), .15*np.sin(np.pi*tau), 0])
        if condition in ['crossing', 'ambiguous_twins']:
            center[:, 0] = [-.78+1.56*tau, .78-1.56*tau]
            center[:, 1] = 0
            if condition == 'ambiguous_twins':
                center[:, 2] = 4
        if condition == 'occlusion':
            center[0, :2] = [-.8+1.6*tau, .12]
            center[1, :2] = [.85, -.4]
        if condition == 'disappearance':
            present[0] = not 12 <= t <= 20
        if condition == 'identity_replacement':
            center = base.copy()
            if 8 <= t < 16:
                present[0] = False
            if t >= 16:
                present[1] = False
                center[0] = base[1]
        # Textured background on a known plane; parallax is physically rendered.
        back_x = (u-K[0, 2])/FOCAL*8+camera[0]
        back_y = (v-K[1, 2])/FOCAL*8+camera[1]
        checker = ((np.floor(back_x/.25)+np.floor(back_y/.25)).astype(int)%2)
        gray = 224+checker*12
        frame = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
        depth = np.full((H, W), 8., dtype=np.float32)
        identity = np.full((H, W), -1, dtype=np.int8)
        colors = np.array([[220, 45, 55], [45, 75, 220]], dtype=float)
        if condition == 'hue_change' and t >= 10:
            colors = np.array([[45, 170, 190], [160, 45, 180]], dtype=float)
        if condition in ['same_hue', 'ambiguous_twins']:
            colors[:] = [65, 95, 205]
        for obj in np.argsort(center[:, 2])[::-1]:
            if not present[obj]:
                continue
            z = center[obj, 2]-camera[2]
            x_world = (u-K[0, 2])/FOCAL*z+camera[0]
            y_world = (v-K[1, 2])/FOCAL*z+camera[1]
            dx, dy = (x_world-center[obj, 0])/RADIUS, (y_world-center[obj, 1])/RADIUS
            circle = dx**2+dy**2 <= 1
            use = circle & (z < depth)
            pattern_obj = 0 if condition == 'ambiguous_twins' else obj
            pattern = (np.floor((dx+1)*(3+pattern_obj))+np.floor((dy+1)*4)).astype(int)%2
            illumination = .62+.32*np.sqrt(np.maximum(0, 1-dx**2-dy**2))
            if condition != 'textureless':
                illumination *= .6+.4*pattern
            colored = np.clip(illumination[..., None]*colors[obj], 0, 255).astype(np.uint8)
            frame[use] = colored[use]
            depth[use] = z
            identity[use] = obj
        if condition in ['occlusion', 'ambiguous_twins']:
            x_occ = (u-K[0, 2])/FOCAL*2.5+camera[0]
            y_occ = (v-K[1, 2])/FOCAL*2.5+camera[1]
            half_width = .13 if condition == 'occlusion' else .38
            occluder = (np.abs(x_occ) <= half_width) & (np.abs(y_occ) <= 1.)
            frame[occluder] = [120, 130, 125]
            depth[occluder] = 2.5
            identity[occluder] = -2
        xyz = np.concatenate([np.column_stack([center[obj, :2]+OFFSETS, np.full(5, center[obj, 2])]) for obj in range(2)])
        cam_points = xyz-camera
        homogeneous = cam_points @ K.T
        xy = homogeneous[:, :2]/homogeneous[:, 2:]
        vis = []
        for query, point in enumerate(xy):
            px, py = np.rint(point).astype(int)
            vis.append(bool(0 <= px < W and 0 <= py < H and identity[py, px] == query//5))
        frames.append(frame); depths.append(depth); positions.append(xyz); projections.append(xy)
        visibility.append(vis); cameras.append(camera); maps.append(identity)
        identifiable.append(np.full(10, not (condition == 'ambiguous_twins' and t >= 12)))
    frames, xyz, xy = np.array(frames), np.array(positions), np.array(projections)
    queries = np.column_stack([np.zeros(10), xy[0]])
    return {'video': frames, 'queries': queries.astype(np.float32)}, {
        'xyz_world': xyz.astype(np.float32), 'xy': xy.astype(np.float32), 'visible': np.array(visibility),
        'identifiable': np.array(identifiable), 'depth': np.array(depths), 'camera_centers': np.array(cameras, dtype=np.float32),
        'intrinsics': K, 'instance_map': np.array(maps), 'radius_world': np.array(RADIUS),
    }


def main():
    output = ROOT / 'data'
    output.mkdir(exist_ok=False)
    manifest = []
    overview = Image.new('RGB', (5*W, 10*(H+20)), 'white')
    draw = ImageDraw.Draw(overview)
    for seed in [910, 1910, 2910]:
        split = 'development' if seed == 910 else 'heldout'
        for row, condition in enumerate(CONDITIONS):
            identifier = f'{condition}_{seed}'
            case = output / identifier
            case.mkdir()
            inputs, truth = render(seed, condition)
            assert inputs['video'].shape == (33, 256, 384, 3) and truth['visible'][0].all(), identifier
            assert np.isfinite(truth['xy']).all() and np.isfinite(truth['xyz_world']).all()
            np.savez_compressed(case / 'input.npz', **inputs)
            np.savez_compressed(case / 'truth.npz', **truth)
            # Frame0 circles are authored from known geometry; setup is allowed query metadata.
            radius = FOCAL*RADIUS/truth['xyz_world'][0, [0, 5], 2]
            setup = {'initial_centers': truth['xy'][0, [0, 5]].tolist(), 'colors_rgb_left_right': [[220,45,55],[45,75,220]],
                     'radius': float(radius.mean()), 'fiducial_centers': [], 'fiducial_half_width': 10,
                     'expected_areas': [int((truth['instance_map'][0] == obj).sum()) for obj in range(2)]}
            (case / 'setup.json').write_text(json.dumps(setup, indent=2)+'\n')
            manifest.append({'case': identifier, 'condition': condition, 'seed': seed, 'split': split,
                             'input': str((case/'input.npz').relative_to(ROOT)), 'input_sha256': sha(case/'input.npz'),
                             'truth': str((case/'truth.npz').relative_to(ROOT)), 'truth_sha256': sha(case/'truth.npz')})
            if seed == 910:
                for column, index in enumerate([0, 8, 16, 24, 32]):
                    x, y = column*W, row*(H+20)
                    overview.paste(Image.fromarray(inputs['video'][index]), (x, y))
                    draw.text((x+4, y+H+2), f'{condition} f{index}', fill='black')
    overview.save(output/'development_overview.jpg', quality=95)
    record = {'cases': manifest, 'renderer_sha256': sha(Path(__file__)), 'protocol_sha256': sha(ROOT/'PROTOCOL.md'),
              'frames_per_case': T, 'fps': 24, 'ground_truth': 'Exact renderer coordinates and z-buffer identity; not human labels.'}
    (ROOT/'data_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'sequences': len(manifest), 'development': 10, 'heldout': 20, 'frames': len(manifest)*T,
                      'renderer_sha256': record['renderer_sha256'], 'protocol_sha256': record['protocol_sha256']}, indent=2))


if __name__ == '__main__':
    main()
