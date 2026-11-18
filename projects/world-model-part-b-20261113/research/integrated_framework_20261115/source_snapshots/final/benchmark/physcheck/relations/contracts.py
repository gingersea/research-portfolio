"""File, prediction and measurement contracts shared by all data adapters."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_file(entry):
    path = Path(entry['path'])
    if sha(path) != entry['sha256']:
        raise ValueError(f'File checksum mismatch: {path}')
    return path


def verify_cache(receipt, input_sha, prediction_sha):
    if receipt.get('status') != 'ok':
        raise ValueError('Cached inference did not succeed')
    bound_input = receipt.get('input_sha256', receipt.get('source_sha256'))
    if bound_input != input_sha:
        raise ValueError('Cache input mismatch')
    bound_prediction = receipt.get('prediction_sha256', receipt.get('depth_sha256'))
    if bound_prediction != prediction_sha:
        raise ValueError('Cache prediction mismatch')
    if receipt.get('future_truth_read', False) or receipt.get('reference_labels_read', False):
        raise ValueError('Cached inference used evaluation labels')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def lift_points(xy, depth, k):
    xy = np.asarray(xy)
    if xy.ndim != 3 or xy.shape[-1] != 2 or depth.ndim != 3 or len(xy) != len(depth):
        raise ValueError('Incompatible XY/depth dimensions')
    if k.shape != (3, 3) or not np.isfinite(k).all() or np.linalg.det(k) == 0:
        raise ValueError('Invalid camera matrix')
    height, width = depth.shape[1:]
    valid = np.isfinite(xy).all(axis=-1) & (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
    index = np.rint(np.where(np.isfinite(xy), xy, 0)).astype(int)
    index[..., 0] = index[..., 0].clip(0, width - 1)
    index[..., 1] = index[..., 1].clip(0, height - 1)
    z = depth[np.arange(len(depth))[:, None], index[..., 1], index[..., 0]]
    valid &= np.isfinite(z) & (z > 0)
    ray = np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1) @ np.linalg.inv(k).T
    xyz = ray * z[..., None]
    xyz[~valid] = np.nan
    return xyz, valid


def relation_decision(measurement, admissible):
    if not admissible or 'score' not in measurement:
        return {'verdict': 'ABSTAIN', 'score': None,
                'reason': 'measurement_assumptions_or_evidence_unavailable',
                'instrument_reason': measurement.get('reason')}
    return {key: measurement[key] for key in ['verdict', 'score', 'reason'] if key in measurement}
