"""New exact controls and prediction-free real-I2V reference images."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    previous=ROOT.parent/'observation_gate_20261115'
    renderer=load('old_renderer',ROOT.parent/'tracking_validation_20261115/render_controls.py')
    textures=load('old_textures',previous/'new_controls.py')
    prior=[]
    for path in [previous/'data_manifest.json',ROOT.parent/'tracking_validation_20261115/data_manifest_v2.json']:
        prior.extend(r['video_pixel_sha256'] for r in json.loads(path.read_text())['cases'])
    output=ROOT/'data';output.mkdir(exist_ok=False)
    records=[]
    conditions=renderer.CONDITIONS+['long_gap_reappearance','lookalike_distractor']
    for seed in [9101,19101,29101,39101]:
        rng=np.random.default_rng(seed+900)
        width=float(rng.uniform(.15,.21));off=int(rng.integers(6,10));on=int(rng.integers(19,25))
        for condition in conditions:
            original='identity_replacement' if condition=='long_gap_reappearance' else 'clean' if condition=='lookalike_distractor' else condition
            inputs,truth=renderer.render(seed,original,corrected=True)
            if condition=='long_gap_reappearance':
                indices=np.array([min(7,t) if t<off else 8 if t<on else 16+min(16,t-on) for t in range(33)])
                inputs['video']=inputs['video'][indices]
                for key in ['xy','xyz_world','visible','identifiable','depth','camera_centers','instance_map']:
                    truth[key]=truth[key][indices]
            if condition=='occlusion':
                u=np.arange(384)[None,:];x=(u-truth['intrinsics'][0,2])/320*2.5
                expanded=np.broadcast_to(abs(x)<=width,(256,384))
                for t in range(33):
                    inputs['video'][t,expanded]=[120,130,125];truth['depth'][t,expanded]=2.5
                    truth['instance_map'][t,expanded]=-2
                    for q,point in enumerate(truth['xy'][t]):
                        xq,yq=np.rint(point).astype(int)
                        truth['visible'][t,q]=truth['instance_map'][t,yq,xq]==q//5
            inputs['video']=textures.recolor(inputs,truth,seed+880000,condition)
            if condition=='lookalike_distractor':
                for t in range(8,33):
                    yy,xx=np.where(truth['instance_map'][t]==0)
                    dx=int(round(192-truth['xy'][t,0,0]));dy=int(round(218-truth['xy'][t,0,1]))
                    inputs['video'][t,yy+dy,xx+dx]=inputs['video'][t,yy,xx]
                    truth['depth'][t,yy+dy,xx+dx]=truth['depth'][t,yy,xx]
                    truth['instance_map'][t,yy+dy,xx+dx]=2
            name=f'{condition}_{seed}';folder=output/name;folder.mkdir()
            np.savez_compressed(folder/'input.npz',**inputs)
            ex=np.repeat(np.eye(4,dtype=np.float32)[None],33,axis=0);ex[:,:3,3]=-truth['camera_centers']
            np.savez_compressed(folder/'rgbd_input.npz',depth=truth['depth'],intrinsics=truth['intrinsics'],extrinsics=ex)
            np.savez_compressed(folder/'truth.npz',**{k:truth[k] for k in ['xy','xyz_world','visible','identifiable']})
            radii=(320*.22/truth['xyz_world'][0,[0,5],2]).tolist()
            setup={'radii':radii,'initial_centers':inputs['queries'][[0,5],1:].tolist()}
            (folder/'setup.json').write_text(json.dumps(setup,indent=2)+'\n')
            pixels=hashlib.sha256(inputs['video'].tobytes()).hexdigest()
            assert pixels not in prior
            record={'case':name,'condition':condition,'seed':seed,'split':'development' if seed==9101 else 'heldout',
                    'occluder_half_width_m':width if condition=='occlusion' else None,
                    'disappearance_onset':off if condition=='long_gap_reappearance' else None,
                    'reappearance_frame':on if condition=='long_gap_reappearance' else None,
                    'video_pixel_sha256':pixels}
            for key,file in [('input','input.npz'),('truth','truth.npz'),('rgbd_input','rgbd_input.npz')]:
                record[key]=str((folder/file).relative_to(ROOT))
                record[{'input':'input_sha256','truth':'truth_sha256','rgbd_input':'rgbd_sha256'}[key]]=sha(folder/file)
            records.append(record)
    assert len({r['video_pixel_sha256'] for r in records})==48
    (ROOT/'data_manifest.json').write_text(json.dumps({'cases':records,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
        'renderer_sha256':sha(Path(__file__)),'recovery_sha256':sha(ROOT/'recovery.py'),
        'old_renderer_sha256':sha(ROOT.parent/'tracking_validation_20261115/render_controls.py')},indent=2)+'\n')
    real=ROOT/'real_i2v';real.mkdir(exist_ok=False)
    original=ROOT.parent/'principia_i2v_20261115'
    real_manifest=[];template=[]
    order=[('B',43),('A',42),('B',42),('A',43)]
    for number,(initial,seed) in enumerate(order,1):
        name=f'C{number}';source=original/f'generation/primary/{initial}/{initial}_seed{seed}_frames.npz'
        with np.load(source) as data:video=data[data.files[0]]
        assert video.shape==(97,512,768,3)
        directory=real/name;directory.mkdir()
        sheet=Image.new('RGB',(1152,3*276),'white');draw=ImageDraw.Draw(sheet)
        for i,t in enumerate(range(0,97,12)):
            im=Image.fromarray(video[t]);im.save(directory/f'frame_{t:03d}.png')
            x,y=(i%3)*384,(i//3)*276
            sheet.paste(im.resize((384,256)),(x,y));draw.text((x+5,y+258),f'{name} frame{t} source768x512',fill='black')
            for obj in range(2):template.append({'clip':name,'frame':t,'object_id':obj,'visibility':'',
                'identity':'','x':'','y':'','uncertainty_px':'','annotator':'','annotator_type':'human','note':''})
        sheet.save(directory/'blind_contact.png')
        setup=json.loads((original/f'inputs/{initial}/setup.json').read_text())
        real_manifest.append({'clip':name,'initial':initial,'seed':seed,'source':str(source),
            'source_sha256':sha(source),'keyframes':list(range(0,97,12)),
            'initial_centers':setup['initial_centers'],'radii':[32.,32.],
            'reference_scope':'new prediction-blind AI visual reference; previous qualitative reports already read'})
    (ROOT/'real_manifest.json').write_text(json.dumps(real_manifest,indent=2)+'\n')
    with (real/'human_annotation_template.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(template[0]));writer.writeheader();writer.writerows(template)
    (real/'annotation_status.json').write_text(json.dumps({'human_labels_received':False,'pending_rows':len(template),
        'AI_reference_is_not_human':True,'predictions_shown_before_reference':False},indent=2)+'\n')
    print(json.dumps({'controls':48,'development':12,'heldout':36,'unique':48,'overlap_prior70':0,'real_i2v':4,'reference_frames':36,'human_rows':72},indent=2))


if __name__=='__main__':main()
