"""Run unchanged official helper ASTs and retain reproducible semantic checks."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Optional

import numpy as np


def load_helpers(source):
    expected_revision = '893e003804f0994ea6467b5a756ab657b82d4df9'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != expected_revision:
        raise ValueError('Unexpected official source revision')
    namespace = {'np': np, 'Optional': Optional, 'Point': tuple[float, float]}
    provenance = {'revision': revision, 'files': {}}
    groups = {
        'src/principia/invariance_score.py': {'compute_delta_phi', 'score_from_delta', 'ratio_score', 'compute_score'},
        'src/principia/scenarios/restitution.py': {'find_impact_frame', 'find_rebound', 'ratio', 'measure_ball'},
    }
    namespace['IMPACT_DY_THRESHOLD'] = 1
    for relative, selected in groups.items():
        path = source / relative
        data = path.read_bytes()
        expected_blob = subprocess.check_output(['git', 'ls-tree', revision, '--', relative], cwd=source, text=True).split()[2]
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected_blob:
            raise ValueError('Official source bytes changed')
        tree = ast.parse(data)
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        if {node.name for node in nodes} != selected:
            raise ValueError('Missing official helper')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        provenance['files'][relative] = {'git_blob': actual, 'sha256': hashlib.sha256(data).hexdigest(), 'functions': sorted(selected)}
    return namespace, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not replace evidence')
    helper, provenance = load_helpers(args.source)
    heights = [2., 1.]
    gravities = [4., 1.]
    times = [math.sqrt(2 * h / g) for h, g in zip(heights, gravities)]
    native_h = helper['ratio'](*heights)
    native_t = helper['ratio'](*times)**2
    native = helper['ratio_score'](native_h, native_t)
    directed = helper['ratio_score'](heights[0] / heights[1], (times[0] / times[1])**2)
    assert np.isclose(native, 1) and np.isclose(directed, .4)
    centers = [(0., y) for y in [1., 2., 5., 8., 10., 10.]]
    contacts = {str(threshold): helper['find_impact_frame'](centers, threshold) for threshold in [12., 10., 9.9]}
    assert contacts == {'12.0': None, '10.0': None, '9.9': 4}
    origin = []
    for common_delay in [0, 30]:
        tracks = []
        for height in [100., 200.]:
            points = [(0., 220 - height + min(height, .5 * 200 * (max(0, i - common_delay) / 30)**2)) for i in range(110)]
            tracks.append(helper['measure_ball'](points, 219.9, 30))
        h = helper['ratio'](*(track['height'] for track in tracks))
        t = helper['ratio'](*(track['time_of_flight'] for track in tracks))**2
        origin.append({'common_delay_frames': common_delay, 'native_frame_zero_score': helper['ratio_score'](h, t), 'measured_impact_frames': [track['impact_frame'] for track in tracks], 'true_gravity_equal': True})
    result = {'official_source': provenance, 'ordering_counterexample': {'heights': heights, 'gravity': gravities, 'flight_times': times, 'native_unordered_score': native, 'directed_score': directed}, 'strict_ground_threshold_counterexample': {'physical_floor': 12, 'radius': 2, 'stationary_center': 10, 'official_contact_by_threshold': contacts}, 'frame_zero_origin_diagnostic': origin, 'scope': 'Verified implementation properties and analytic fixtures, not an evaluation of all Principia results or a claim that all published annotations use a physical floor line.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
