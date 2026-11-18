"""Prepare fixed, known-geometry initial images and separate instrument controls."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

STUDY = Path(__file__).resolve().parent
WIDTH, HEIGHT, RADIUS, FLOOR, FPS, FRAMES = 768, 512, 32, 464, 24, 97
COLORS = [[225, 35, 45], [35, 65, 225]]
XS = [224, 544]
MARKERS = [[40, 32], [728, 32], [40, 488], [728, 488]]
PROMPT = ('A fixed-camera, real-time scientific demonstration of the setup in the image. '
          'Two equal-sized balls of the same material are initially held at rest at the shown heights above the same horizontal floor. '
          'At the start both balls are released simultaneously. The camera and green reference markers stay fixed. '
          'Only the two existing balls, floor and background are present. The complete motion of both balls remains in view.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(centers):
    frame = np.full((HEIGHT, WIDTH, 3), 242, dtype=np.uint8)
    frame[FLOOR + 1:] = 225
    cv2.line(frame, (20, FLOOR), (WIDTH - 20, FLOOR), (70, 70, 70), 2)
    for x, y in MARKERS:
        cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (25, 180, 35), -1)
    masks = []
    for x, y, color in zip(XS, centers, COLORS):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.circle(mask, (x, round(y)), RADIUS, 1, -1)
        ys, xs = np.where(mask)
        dx, dy = (xs - x) / RADIUS, (ys - round(y)) / RADIUS
        z = np.sqrt(np.maximum(0, 1 - dx**2 - dy**2))
        illumination = np.clip(.70 + .22 * z - .08 * dx - .07 * dy, .55, 1)
        frame[ys, xs] = np.rint(illumination[:, None] * np.array(color)).astype(np.uint8)
        masks.append(mask)
    return frame, masks


def main():
    root = STUDY / 'inputs'
    if root.exists():
        raise FileExistsError('Refuse to replace fixed inputs')
    root.mkdir()
    source = STUDY.parent / 'principia_gravity_20261114/measure.py'
    old_manifest = json.loads((STUDY.parent / 'principia_gravity_20261114/source_snapshots/heldout_v2/manifest.json').read_text())
    assert sha(source) == old_manifest['source_sha256']['measure.py']
    spec = importlib.util.spec_from_file_location('frozen_gravity_measure', source)
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    setups, controls = [], []
    for identifier, heights in [('A', [160., 288.]), ('B', [288., 160.])]:
        directory = root / identifier
        directory.mkdir()
        centers = [FLOOR - RADIUS - height for height in heights]
        initial, masks = draw(centers)
        image_path = directory / 'initial.png'
        Image.fromarray(initial).save(image_path)
        np.savez_compressed(directory / 'masks.npz', left=masks[0], right=masks[1])
        setup = {'initial_id': identifier, 'image': str(image_path.relative_to(STUDY)), 'image_sha256': sha(image_path), 'width': WIDTH, 'height': HEIGHT, 'radius': RADIUS, 'floor_y': FLOOR, 'colors_rgb_left_right': COLORS, 'initial_centers': [[x, y] for x, y in zip(XS, centers)], 'fiducial_centers': MARKERS, 'fiducial_half_width': 10, 'drop_heights_px': heights, 'prompt': PROMPT, 'seeds': [42, 43], 'scope': 'Authored procedural shaded spheres; not a real Principia photograph.'}
        (directory / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
        setups.append(setup)
        for condition, gravity in [('equal', [420., 420.]), ('unequal', [420., 1680.])]:
            frames = []
            for index in range(FRAMES):
                t = index / FPS
                ys = [FLOOR - RADIUS - height + min(height, .5 * g * t**2) for height, g in zip(heights, gravity)]
                frames.append(draw(ys)[0])
            array = np.stack(frames)
            target = directory / f'instrument_{condition}.mkv'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-level', '3', '-threads', '1', str(target)], input=array.tobytes(), check=True)
            result = measurement.measure_video(target, {**setup, 'fps': FPS, 'frames': FRAMES})
            expected = 'PASS' if condition == 'equal' else 'FAIL'
            controls.append({'initial_id': identifier, 'condition': condition, 'expected': expected, 'observed': result['verdict'], 'reason': result['reason'], 'score': result.get('score'), 'video': str(target.relative_to(STUDY)), 'sha256': sha(target), 'frozen_measure_source_sha256': sha(source)})
    (STUDY / 'input_manifest.json').write_text(json.dumps(setups, indent=2) + '\n')
    passed = all(row['expected'] == row['observed'] for row in controls)
    (STUDY / 'instrument_preflight.json').write_text(json.dumps({'passed': passed, 'controls': controls, 'source_sha256': sha(Path(__file__)), 'scope': 'New geometry/FPS and shaded-color control replay only; no generated-video accuracy claim.'}, indent=2) + '\n')
    print(json.dumps({'inputs': len(setups), 'preflight_passed': passed, 'controls': [{key: row[key] for key in ['initial_id', 'condition', 'expected', 'observed', 'score']} for row in controls]}, indent=2))


if __name__ == '__main__':
    main()
