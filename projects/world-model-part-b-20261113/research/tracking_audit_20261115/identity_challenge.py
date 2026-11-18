"""Adversarial known-identity replacement, outside the initial control family."""
import json
from pathlib import Path

import cv2
import numpy as np

from candidate import track_observed
from run_correction import control_frames
from diagnose import sha

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    output = ROOT / 'identity_challenge_v1'
    output.mkdir(exist_ok=False)
    frames, setup, _ = control_frames(710, 'clean')
    # Known source identities: frame0 has A(left) and B(right), frame1 only B;
    # from frame2 onward B is absent, A is drawn at B's previous position.
    # Geometry/appearance observations alone cannot distinguish this from B
    # persisting and changing appearance. The truth below is from our renderer.
    first = frames[0].copy()
    background = np.repeat(np.median(first, axis=1, keepdims=True), first.shape[1], axis=1).astype(np.uint8)
    for x, y in setup['fiducial_centers']:
        cv2.rectangle(background, (x-10, y-10), (x+10, y+10), (25, 180, 35), -1)
    images = [first]
    right = tuple(setup['initial_centers'][1])
    for t in range(1, 6):
        image = background.copy()
        cv2.circle(image, right, 32, [35, 65, 225] if t == 1 else [225, 35, 45], -1)
        images.append(image)
    pixels = np.array(images)
    tracks = track_observed(pixels, setup, first)
    false_right_identity = sum(row[1]['observed'] for row in tracks[2:])
    np.savez_compressed(output / 'frames.npz', frames=pixels)
    result = {'candidate_sha256': sha(ROOT / 'candidate.py'), 'frames': 6,
              'known_renderer_identity': [['A', 'B'], ['B']] + [['A']]*4,
              'right_B_absent_frames': 4, 'wrong_B_identity_claims': false_right_identity,
              'passed_identity_safety': false_right_identity == 0, 'tracks': tracks,
              'interpretation': 'Prototype can observe a current blob but cannot certify persistent material identity under disappearance/replacement. Do not promote it to physics scoring.'}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'tracks'}, indent=2))
