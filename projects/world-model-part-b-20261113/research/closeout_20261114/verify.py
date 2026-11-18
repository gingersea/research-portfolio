"""Read-only independent replay of retained media/data; no model/API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check_hashes(directory: Path) -> int:
    checksums=read(directory/'checksums.json')
    for name,expected in checksums.items():
        path=(directory/name).resolve()
        assert path.is_relative_to(directory.resolve()),name
        assert digest(path)==expected,name
    return len(checksums)


def decode(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',type=Path)
    args=parser.parse_args()
    old=ROOT/'research/part_b_20261113T114040Z'
    close=ROOT/'research/closeout_20261114/results'
    pilot=ROOT/'research/action_world_stage0_20261114/results'
    manifest=ROOT/'research/closeout_20261114/package_manifest.json'
    if manifest.exists():
        for name,expected in read(manifest)['source_and_summary_sha256'].items():
            path=(ROOT/name).resolve()
            assert path.is_relative_to(ROOT.resolve()),name
            assert digest(path)==expected,name
    old_count=check_hashes(old) if old.exists() else None
    counts={str(p.relative_to(ROOT)):check_hashes(p) for p in [close,pilot]}
    for name,expected in read(close/'provenance.json')['source_sha256'].items():
        assert digest(ROOT/name)==expected,name
    p=read(pilot/'provenance.json')
    assert digest(pilot.parent/'experiment.py')==p['source_sha256']
    assert digest(ROOT/'docs/protocols/action_world_stage0_20261114.md')==p['protocol_sha256']
    coverage=read(close/'coverage.json')
    assert len(coverage)==50
    for row in coverage:
        n,k,d=row['frames'],row['samples'],row['duration_frames']
        selected=sorted({round(i*(n-1)/(k-1)) for i in range(k)})
        assert row['sampled_indices']==selected
        mask=np.zeros(n,dtype=int);mask[selected]=1
        prefix=np.r_[0,np.cumsum(mask)]
        hits=np.flatnonzero(prefix[d:]-prefix[:-d]>0).tolist()
        assert hits==row['visible_start_indices']
        assert len(hits)==row['visible_windows']
        assert row['exposure_fraction']==len(hits)/(n-d+1)
    frame_count=0;pair_count=0
    for pair in read(close/'pairs.json'):
        videos=[decode(close/name) for name in pair['video_paths']]
        assert [digest(close/name) for name in pair['video_paths']]==pair['video_sha256']
        different=np.flatnonzero(np.any(videos[0]!=videos[1],axis=(1,2,3))).tolist()
        assert different==[pair['event_frame']]
        grid_paths=[close/'artifacts/frames'/f'{pair["pair_id"]}_{label}'/'grid.png'
                    for label in ['control','intervention']]
        assert grid_paths[0].read_bytes()==grid_paths[1].read_bytes()
        assert digest(grid_paths[0])==pair['grid_sha256']
        for label,video in zip(['control','intervention'],videos):
            assert len(video)==pair['frames']
            frame_count+=len(video)
            for idx in pair['sampled_indices']:
                png=cv2.imread(str(close/'artifacts/frames'/f'{pair["pair_id"]}_{label}'/f'frame_{idx:04d}.png'))
                assert np.array_equal(cv2.cvtColor(png,cv2.COLOR_BGR2RGB),video[idx])
        assert np.array_equal(videos[0][pair['sampled_indices']],videos[1][pair['sampled_indices']])
        pair_count+=1
    test=np.load(pilot/'test_data.npz',allow_pickle=False)
    true_states=test['states'];actions=test['actions']
    assert true_states.shape==(128,31,4) and actions.shape==(128,30,2)
    assert np.all(np.abs(actions)==1)
    train_actions=set()
    for seed in [7,17,27]:
        train=np.load(pilot/f'train_seed_{seed}.npz',allow_pickle=False)
        train_actions.update(map(tuple,train['actions'].reshape(-1,2)))
        assert train['states'].shape==(256,17,4)
    assert train_actions.isdisjoint(set(map(tuple,actions.reshape(-1,2))))
    summaries=read(pilot/'prediction_summary.json')
    assert len(summaries)==6
    for row in summaries:
        weights=np.load(pilot/f'{row["method"]}_seed{row["train_seed"]}.npz',allow_pickle=False)
        predicted=true_states[:,0].copy()
        for step in range(1,31):
            predicted=predicted@weights['A'].T+actions[:,step-1]@weights['B'].T+weights['bias']
            if step in [1,10,30]:
                delta=predicted-true_states[:,step]
                for metric,values in [('state',delta),('position',delta[:,:2]),('velocity',delta[:,2:])]:
                    actual=float(np.sqrt(np.mean(values**2)))
                    assert np.isclose(actual,row[f'{metric}_rmse_h{step}'],rtol=1e-10,atol=1e-14)
    cases=np.load(pilot/'planning_cases.npz',allow_pickle=False)
    episodes=read(pilot/'planning_episodes.json')
    assert len(episodes)==448
    grouped=Counter();successes=Counter();seen=set()
    for episode in episodes:
        key=(episode['model'],episode['episode']);assert key not in seen;seen.add(key)
        i=episode['episode'];states=np.array(episode['states']);acts=np.array(episode['actions'])
        assert states.shape==(33,4) and acts.shape==(32,2) and np.isfinite(states).all()
        assert (np.abs(acts)<=1).all()
        assert np.array_equal(states[0],cases['initial'][i])
        assert np.array_equal(episode['goal'],cases['goals'][i])
        velocity=.92*states[:-1,2:]+.2*acts
        assert np.allclose(states[1:,2:],velocity,rtol=0,atol=1e-14)
        assert np.allclose(states[1:,:2],states[:-1,:2]+.2*velocity,rtol=0,atol=1e-14)
        distance=float(np.linalg.norm(states[-1,:2]-cases['goals'][i]))
        assert np.isclose(distance,episode['final_distance'],rtol=0,atol=1e-14)
        assert type(episode['success']) is bool and episode['success']==(distance<=.2)
        grouped[episode['model']]+=1;successes[episode['model']]+=episode['success']
    assert len(grouped)==7 and set(grouped.values())=={64}
    for row in read(pilot/'planning_summary.json'):
        assert row['successes']==successes[row['model']] and row['episodes']==grouped[row['model']]
    observed={'passed':True,'new_result_checksums':counts,'old_part_b_checksums_preserved':old_count,
              'enumerated_conditions_recomputed':50,'media_pairs_verified':pair_count,
              'decoded_new_synthetic_frames':frame_count,'prediction_models_replayed':len(summaries),
              'planning_runs_replayed':len(episodes),'paired_planning_cases':64,
              'protocol_and_executed_source_hashes_match':True}
    if args.record:
        if args.record.exists():
            raise ValueError('Refusing to overwrite a verification receipt')
        args.record.write_text(json.dumps(observed,indent=2)+'\n')
    print(json.dumps(observed,indent=2))


if __name__=='__main__':main()
