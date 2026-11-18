"""Fresh controls after retiring all V1 results to development."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'identity_recovery_20261115'),str(ROOT.parent/'observation_gate_20261115')]
from fetch_depth import sha


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    renderer=load('geometry',ROOT.parent/'tracking_validation_20261115/render_controls.py')
    textures=load('texture',ROOT.parent/'observation_gate_20261115/new_controls.py')
    prior=set()
    for directory,file in [('tracking_validation_20261115','data_manifest_v2.json'),('observation_gate_20261115','data_manifest.json'),('identity_recovery_20261115','data_manifest.json')]:
        prior|={r['video_pixel_sha256'] for r in json.loads((ROOT.parent/directory/file).read_text())['cases']}
    output=ROOT/'data';output.mkdir(exist_ok=False);rows=[]
    for seed in [91201,191201,291201,391201]:
        rng=np.random.default_rng(seed+900);width=float(rng.uniform(.15,.21));off=int(rng.integers(6,10));on=int(rng.integers(19,25))
        for condition in renderer.CONDITIONS+['long_gap_reappearance','lookalike_distractor']:
            original='identity_replacement' if condition=='long_gap_reappearance' else 'clean' if condition=='lookalike_distractor' else condition
            inputs,truth=renderer.render(seed,original,corrected=True)
            if condition=='long_gap_reappearance':
                idx=np.array([min(7,t) if t<off else 8 if t<on else 16+min(16,t-on) for t in range(33)])
                inputs['video']=inputs['video'][idx]
                for key in ['xy','xyz_world','visible','identifiable','depth','camera_centers','instance_map']:truth[key]=truth[key][idx]
            if condition=='occlusion':
                u=np.arange(384)[None,:];expanded=np.broadcast_to(abs((u-truth['intrinsics'][0,2])/320*2.5)<=width,(256,384))
                for t in range(33):
                    inputs['video'][t,expanded]=[120,130,125];truth['depth'][t,expanded]=2.5;truth['instance_map'][t,expanded]=-2
                    for q,p in enumerate(truth['xy'][t]):
                        x,y=np.rint(p).astype(int);truth['visible'][t,q]=truth['instance_map'][t,y,x]==q//5
            inputs['video']=textures.recolor(inputs,truth,seed+880000,condition)
            if condition=='lookalike_distractor':
                for t in range(8,33):
                    yy,xx=np.where(truth['instance_map'][t]==0)
                    dx=int(round(192-truth['xy'][t,0,0]));dy=int(round(218-truth['xy'][t,0,1]))
                    inputs['video'][t,yy+dy,xx+dx]=inputs['video'][t,yy,xx]
                    truth['depth'][t,yy+dy,xx+dx]=truth['depth'][t,yy,xx];truth['instance_map'][t,yy+dy,xx+dx]=2
            name=f'{condition}_{seed}';folder=output/name;folder.mkdir()
            np.savez_compressed(folder/'input.npz',**inputs)
            ex=np.repeat(np.eye(4,dtype=np.float32)[None],33,axis=0);ex[:,:3,3]=-truth['camera_centers']
            np.savez_compressed(folder/'rgbd_input.npz',depth=truth['depth'],intrinsics=truth['intrinsics'],extrinsics=ex)
            np.savez_compressed(folder/'truth.npz',**{k:truth[k] for k in ['xy','xyz_world','visible','identifiable']})
            (folder/'setup.json').write_text(json.dumps({'radii':(320*.22/truth['xyz_world'][0,[0,5],2]).tolist(),
                'initial_centers':inputs['queries'][[0,5],1:].tolist()},indent=2)+'\n')
            pixels=hashlib.sha256(inputs['video'].tobytes()).hexdigest();assert pixels not in prior
            row={'case':name,'condition':condition,'seed':seed,'split':'development' if seed==91201 else 'heldout',
                 'occluder_half_width_m':width if condition=='occlusion' else None,
                 'reappearance_frame':on if condition=='long_gap_reappearance' else None,'video_pixel_sha256':pixels}
            for key,file,digest in [('input','input.npz','input_sha256'),('truth','truth.npz','truth_sha256'),('rgbd_input','rgbd_input.npz','rgbd_sha256')]:
                row[key]=str((folder/file).relative_to(ROOT));row[digest]=sha(folder/file)
            rows.append(row)
    assert len({r['video_pixel_sha256'] for r in rows})==48
    (ROOT/'data_manifest.json').write_text(json.dumps({'cases':rows,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
        'recovery_sha256':sha(ROOT/'recovery.py'),'source_sha256':sha(Path(__file__)),'overlap_with_prior118':0},indent=2)+'\n')
    print(json.dumps({'cases':48,'heldout':36,'prior_controls_retired':118,'overlap':0}))


if __name__=='__main__':main()
