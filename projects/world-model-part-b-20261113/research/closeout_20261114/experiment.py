"""Finite sampling audit and lossless paired counterexamples; no VLM labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'benchmark'))
from physcheck.config import Config
from physcheck.judge.frames import sample_frames


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def indices(n: int,k: int) -> list[int]:
    return sorted({round(i*(n-1)/(k-1)) for i in range(k)})


def coverage(n: int,k: int,duration: int) -> dict:
    sampled=indices(n,k)
    starts=list(range(n-duration+1))
    hit=[start for start in starts if any(start<=i<start+duration for i in sampled)]
    return {'frames':n,'samples':k,'duration_frames':duration,
            'sampled_indices':sampled,'legal_windows':len(starts),
            'visible_windows':len(hit),'missed_windows':len(starts)-len(hit),
            'exposure_fraction':len(hit)/len(starts),'visible_start_indices':hit,
            'max_unsampled_run':max(b-a-1 for a,b in zip(sampled,sampled[1:]))}


def state(n: int,seed: int) -> list[dict]:
    u=np.linspace(0,1,n)
    return [{'x':float(72+5*seed+40*t),'y':float(28+4*seed+85*t*t),
             'radius':9+seed} for t in u]


def frame_at(s: dict,seed: int) -> np.ndarray:
    image=np.full((180,320,3),(235-5*seed,240,248),dtype=np.uint8)
    cv2.line(image,(0,158),(319,158),(90,90,90),2)
    cv2.circle(image,(round(s['x']),round(s['y'])),s['radius'],
               (210,45+20*seed,35),-1,lineType=cv2.LINE_8)
    return image


def write_video(path: Path,frames: np.ndarray,fps: int) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(path),'w',format='matroska') as container:
        stream=container.add_stream('ffv1',rate=fps)
        stream.width=frames.shape[2];stream.height=frames.shape[1]
        stream.pix_fmt='bgr0';stream.time_base=Fraction(1,fps)
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def decode(path: Path,fps: int) -> np.ndarray:
    with av.open(str(path)) as container:
        assert float(container.streams.video[0].average_rate)==fps
        return np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])


def run_pair(output: Path,n: int,fps: int,kind: str,seed: int,cfg: Config) -> dict:
    pair_id=f'n{n}_{kind}_s{seed}'
    states=state(n,seed)
    baseline=np.stack([frame_at(s,seed) for s in states])
    sampled=indices(n,8)
    left,right=max(zip(sampled,sampled[1:]),key=lambda p:p[1]-p[0])
    event=(left+right)//2
    assert event not in sampled
    changed_state=dict(states[event])
    if kind=='radius':
        changed_state['radius']*=2
    else:
        changed_state['x']+=55
    changed=baseline.copy();changed[event]=frame_at(changed_state,seed)
    paths=[output/'videos'/f'{pair_id}_{label}.mkv' for label in ['control','intervention']]
    source=[baseline,changed]
    sampled_records=[];decoded=[]
    for label,path,rgb in zip(['control','intervention'],paths,source):
        write_video(path,rgb,fps)
        actual=decode(path,fps)
        assert np.array_equal(actual,rgb),f'lossless roundtrip failed: {path}'
        decoded.append(actual)
        result=sample_frames(str(path),8,cfg,f'{pair_id}_{label}')
        assert result['indices']==sampled
        sampled_records.append(result)
    differing=np.flatnonzero(np.any(decoded[0]!=decoded[1],axis=(1,2,3))).tolist()
    assert differing==[event]
    assert digest(paths[0])!=digest(paths[1])
    first,second=sampled_records
    assert all(Path(a).read_bytes()==Path(b).read_bytes()
               for a,b in zip(first['frame_paths'],second['frame_paths']))
    assert Path(first['grid_path']).read_bytes()==Path(second['grid_path']).read_bytes()
    for label,rgb in zip(['control','intervention'],source):
        path=output/'event_frames'/f'{pair_id}_{label}.png'
        path.parent.mkdir(parents=True,exist_ok=True)
        assert cv2.imwrite(str(path),cv2.cvtColor(rgb[event],cv2.COLOR_RGB2BGR))
    return {'pair_id':pair_id,'frames':n,'fps':fps,'seed':seed,'kind':kind,
            'event_frame':event,'sampled_indices':sampled,'differing_decoded_frames':differing,
            'lossless_roundtrip_verified':True,'sampled_pngs_identical':True,
            'grid_bytes_identical':True,'grid_sha256':digest(Path(first['grid_path'])),
            'video_paths':[p.relative_to(output).as_posix() for p in paths],
            'video_sha256':[digest(p) for p in paths],
            'baseline_states':states,'intervention_state':changed_state,
            'ground_truth_scope':'Violation of this prescribed rigid-ball ballistic world; not human labels or general generated-video truth.'}


def render_report(output: Path,rows: list[dict],pairs: list[dict]) -> None:
    lines=['# Closeout: what eight sampled frames can miss','',
           'This is a controlled methodology audit, not a video-model ranking or VLM accuracy measurement.',
           'Protocol was fixed locally before creating this run. Existing project outcomes were already known; the study is exploratory.',
           '', '## Exact window exposure','',
           '| Video frames | Samples | Event duration (frames) | Visible / possible windows | Exposure |',
           '|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['samples'] in [8,16,32] and row['duration_frames'] in [1,4,8]:
            lines.append(f'| {row["frames"]} | {row["samples"]} | {row["duration_frames"]} | {row["visible_windows"]}/{row["legal_windows"]} | {row["exposure_fraction"]:.2%} |')
    lines+=['','Exact finite enumeration over legal start positions; no inferential confidence interval.',
            'Exposure assumes uniformly weighted discrete starts and is only a necessary visibility condition, not semantic detection accuracy.',
            'No claim is made that actual generator errors follow this timing distribution.', '',
            '## Lossless counterexamples','',
            f'{len(pairs)} pairs / {2*len(pairs)} FFV1 videos: source RGB equals decoded RGB; each pair differs at exactly one designated unsampled frame.',
            f'All {len(pairs)} pairs have byte-identical eight-frame PNGs and annotated grids, while full-video states and video hashes differ.',
            'Labels describe deviations from a specified fixed-camera rigid-ball world with no external contact/force, not independently human-labeled natural physics.',
            'If opposite classes have equal priors and the classifier receives only identical images without label leakage, expected image-only discrimination is 50%. This is a conditional information bound, not an observed VLM result.',
            'Dense sampling reveals the programmed differences; it does not by itself certify physical interpretation.', '',
            '## Closeout decision','',
            'The old pipeline is closed as a feasibility/methodology study with explicit negative outcomes. Its Part A SI estimates and Part B calibrated ranking remain unavailable.',
            'Do not spend further generator runs merely to force the old calibration gate to pass. Reuse code, frozen artifacts and controlled audit tools for a differently scoped question.',
            'This experiment demonstrates a known sparse-observation limitation in this specific pipeline; it does not establish a novel general theorem or a publishable research contribution on its own.', '',
            '## Reproduction','',
            '`python research/closeout_20261114/experiment.py --output NEW_DIRECTORY`',
            'The output must not exist. The current repository sampler is invoked without modifying it. No model generation, paid API call, human label or VLM judgment is fabricated.',
            'See coverage.json for all50 conditions, pairs.json for states/interventions and verification.json for observed checks.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=STUDY/'results')
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory; never overwrite a completed or partial run.')
    output.mkdir(parents=True)
    source_paths=[Path(__file__),ROOT/'benchmark/physcheck/judge/frames.py',
                  ROOT/'docs/protocols/world_model_closeout_20261114.md']
    dump(output/'provenance.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
         'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_paths},
         'python':platform.python_version(),'av':av.__version__,'numpy':np.__version__,
         'opencv':cv2.__version__,'scope':'CPU-only synthetic measurement audit; not VLM calibration or human truth.'})
    rows=[coverage(n,k,d) for n in [49,81] for k in [4,8,16,32,n] for d in [1,2,4,8,16]]
    assert len(rows)==50
    assert all(r['visible_windows']==r['legal_windows'] for r in rows if r['samples']==r['frames'])
    assert all(r['visible_windows']==r['samples'] for r in rows if r['duration_frames']==1)
    dump(output/'coverage.json',rows)
    cfg=Config(ROOT,output/'artifacts',output/'unused_models',output/'unused_results')
    pairs=[run_pair(output,n,fps,kind,seed,cfg) for n,fps in [(49,8),(81,16)]
           for kind in ['radius','position'] for seed in [0,1,2]]
    dump(output/'pairs.json',pairs)
    verification={'passed':True,'enumerated_conditions':len(rows),'pairs':len(pairs),
                  'videos':2*len(pairs),'decoded_frames':sum(2*p['frames'] for p in pairs),
                  'identical_eight_frame_pairs':sum(p['grid_bytes_identical'] for p in pairs),
                  'lossless_roundtrip_verified':all(p['lossless_roundtrip_verified'] for p in pairs),
                  'vlm_calls':0,'human_labels_created':0}
    dump(output/'verification.json',verification)
    render_report(output,rows,pairs)
    dump(output/'checksums.json',{p.relative_to(output).as_posix():digest(p)
                                 for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
