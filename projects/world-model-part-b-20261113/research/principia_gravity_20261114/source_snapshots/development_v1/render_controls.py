"""Analytic paired-gravity controls; private truth is separate from public media."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

CONDITIONS = ['equal_gravity', 'unequal_gravity', 'release_delay', 'occlusion', 'camera_pan', 'truncated']
WIDTH, HEIGHT, FPS, FRAMES = 512, 384, 30, 96


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_scene(seed, split, output):
    rng = np.random.default_rng(seed)
    radius = int(rng.integers(12, 17))
    floor = 352
    heights = [float(rng.uniform(110, 160)), float(rng.uniform(200, 260))]
    if rng.random() < .5:
        heights.reverse()
    gravity = float(rng.uniform(280, 420))
    colors = [[225, 35, 45], [35, 65, 225]]
    if rng.random() < .5:
        colors.reverse()
    x_positions = [160, 352]
    public, private = [], []
    for condition in CONDITIONS:
        identifier = 'c_' + hashlib.sha256(f'paired-gravity-v1|{seed}|{condition}'.encode()).hexdigest()[:18]
        directory = output / 'public' / identifier
        directory.mkdir(parents=True, exist_ok=False)
        accelerations = [gravity, gravity * (4 if condition == 'unequal_gravity' else 1)]
        releases = [0, 8 if condition == 'release_delay' else 0]
        count = 12 if condition == 'truncated' else FRAMES
        times = np.arange(count) / FPS
        y = np.empty((count, 2))
        impacts = []
        continuous_impacts = []
        for ball in range(2):
            age = np.maximum(times - releases[ball] / FPS, 0)
            y[:, ball] = floor - radius - heights[ball] + np.minimum(.5 * accelerations[ball] * age**2, heights[ball])
            contact = releases[ball] + math.sqrt(2 * heights[ball] / accelerations[ball]) * FPS
            continuous_impacts.append(contact)
            impacts.append(math.ceil(contact))
        frames = np.full((count, HEIGHT, WIDTH, 3), 246, dtype=np.uint8)
        for index, frame in enumerate(frames):
            cv2.line(frame, (20, floor), (492, floor), (70, 70, 70), 2)
            cv2.line(frame, (256, 45), (256, floor), (215, 215, 215), 1)
            for marker in [(30, 25), (482, 25), (30, 375), (482, 375)]:
                cv2.rectangle(frame, (marker[0] - 4, marker[1] - 4), (marker[0] + 4, marker[1] + 4), (25, 180, 35), -1)
            for ball in range(2):
                cv2.circle(frame, (x_positions[ball], round(y[index, ball])), radius, colors[ball], -1)
            if condition == 'occlusion' and 4 <= index <= 9:
                center = (x_positions[0], round(y[index, 0]))
                cv2.rectangle(frame, (center[0] - radius - 6, center[1] - radius - 6), (center[0] + radius + 6, center[1] + radius + 6), (110, 110, 110), -1)
            if condition == 'camera_pan':
                dx = round(12 * np.sin(index / 22))
                frames[index] = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, 0]]), (WIDTH, HEIGHT), borderValue=(246, 246, 246))
        paths = {}
        for codec, extension in [('ffv1', 'mkv'), ('h264', 'mp4')]:
            target = directory / f'clip.{extension}'
            command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an']
            command += ['-c:v', 'ffv1', '-level', '3', '-threads', '1'] if codec == 'ffv1' else ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-threads', '1']
            subprocess.run(command + [str(target)], input=frames.tobytes(), check=True)
            paths[codec] = {'path': str(target.relative_to(output)), 'sha256': digest(target)}
        setup = {'case_id': identifier, 'fps': FPS, 'width': WIDTH, 'height': HEIGHT, 'frames': count, 'floor_y': floor, 'radius': radius, 'colors_rgb_left_right': colors, 'setup': 'Two equal-radius balls of the same material in one image plane, with the same floor. Verify synchronized release and camera stability from the observed sequence. No bouncing is considered.'}
        (directory / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
        public.append({'case_id': identifier, 'split': split, 'setup_path': str((directory / 'setup.json').relative_to(output)), 'videos': paths})
        expected = 'PASS' if condition == 'equal_gravity' else 'FAIL' if condition == 'unequal_gravity' else 'ABSTAIN'
        h_ratio = heights[0] / heights[1]
        t_ratio_squared = (math.sqrt(2 * heights[0] / accelerations[0]) / math.sqrt(2 * heights[1] / accelerations[1]))**2
        private.append({'case_id': identifier, 'scene_seed': seed, 'split': split, 'condition': condition, 'expected': expected, 'heights_px': heights, 'accelerations_px_s2': accelerations, 'release_frames': releases, 'contact_frames_ceil': impacts, 'contact_frames_continuous': continuous_impacts, 'height_ratio': h_ratio, 'time_ratio_squared': t_ratio_squared, 'symmetric_score': 1 - abs(h_ratio - t_ratio_squared) / (h_ratio + t_ratio_squared), 'centers_y_unrounded': y.tolist()})
    return public, private


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'heldout'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / f'{args.split}_public_index.json'
    truth_path = args.output / 'private' / f'{args.split}_truth.json'
    if index_path.exists() or truth_path.exists():
        raise FileExistsError('Refusing to replace an existing dataset')
    seeds = list(range(71000, 71004)) if args.split == 'development' else list(range(81000, 81024))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda seed: render_scene(seed, args.split, args.output), seeds))
    public = [row for result in results for row in result[0]]
    private = [row for result in results for row in result[1]]
    truth_path.parent.mkdir(exist_ok=True)
    index_path.write_text(json.dumps(public, indent=2) + '\n')
    truth_path.write_text(json.dumps(private, indent=2) + '\n')
    print(json.dumps({'split': args.split, 'scenes': len(seeds), 'cases': len(public), 'encoded_clips': 2 * len(public), 'source_sha256': digest(Path(__file__))}))


if __name__ == '__main__':
    main()
