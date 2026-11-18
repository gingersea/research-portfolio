"""New textures/seeds over frozen physical geometry, with separate RGB-D inputs/truth."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from fetch_tapip import sha

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'tracking_validation_20261115'
spec=importlib.util.spec_from_file_location('old_geometry',OLD/'render_controls.py')
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


def recolor(inputs,truth,seed,condition):
    rng=np.random.default_rng(seed+170001)
    textures=[]
    for obj in range(2):
        raw=rng.random((16,16)).astype(np.float32)
        texture=cv2.GaussianBlur(cv2.resize(raw,(64,64),interpolation=cv2.INTER_CUBIC),(0,0),.7)
        texture=np.clip((texture-texture.min())/(texture.max()-texture.min()),0,1)
        textures.append(.45+.55*texture)
    if condition=='ambiguous_twins':textures[1]=textures[0].copy()
    video=inputs['video'].copy()
    height,width=video.shape[1:3]
    u,v=np.meshgrid(np.arange(width),np.arange(height))
    K=truth['intrinsics']
    for t,frame in enumerate(video):
        colors=np.array([[220,45,55],[45,75,220]])
        if condition=='hue_change' and t>=10:colors=np.array([[45,170,190],[160,45,180]])
        if condition in ['same_hue','ambiguous_twins']:colors[:]=[65,95,205]
        for obj in range(2):
            use=truth['instance_map'][t]==obj
            center=truth['xyz_world'][t,obj*5]
            camera=truth['camera_centers'][t]
            depth=center[2]-camera[2]
            dx=((u-K[0,2])/K[0,0]*depth+camera[0]-center[0])/.22
            dy=((v-K[1,2])/K[1,1]*depth+camera[1]-center[1])/.22
            texture=cv2.remap(textures[obj],((dx+1)*31.5).astype(np.float32),((dy+1)*31.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            illumination=.62+.32*np.sqrt(np.maximum(0,1-dx**2-dy**2))
            if condition!='textureless':illumination*=texture
            colored=np.clip(illumination[...,None]*colors[obj],0,255).astype(np.uint8)
            frame[use]=colored[use]
    return video


def main():
    output=ROOT/'data'
    output.mkdir(exist_ok=False)
    manifest=[]
    prior=json.loads((OLD/'data_manifest_v2.json').read_text())
    old_pixels={row['video_pixel_sha256'] for row in prior['cases']}
    overview=Image.new('RGB',(5*384,10*276),'white');draw=ImageDraw.Draw(overview)
    import hashlib
    for seed in [810,1810,2810,3810]:
        split='development' if seed==810 else 'heldout'
        for row,condition in enumerate(geometry.CONDITIONS):
            inputs,truth=geometry.render(seed,condition,corrected=True)
            inputs['video']=recolor(inputs,truth,seed,condition)
            digest=hashlib.sha256(inputs['video'].tobytes()).hexdigest()
            assert digest not in old_pixels
            name=f'{condition}_{seed}';folder=output/name;folder.mkdir()
            np.savez_compressed(folder/'input.npz',**inputs)
            extrinsics=np.repeat(np.eye(4,dtype=np.float32)[None],33,axis=0)
            extrinsics[:,:3,3]=-truth['camera_centers']
            np.savez_compressed(folder/'rgbd_input.npz',depth=truth['depth'],intrinsics=truth['intrinsics'],extrinsics=extrinsics)
            np.savez_compressed(folder/'truth.npz',xy=truth['xy'],xyz_world=truth['xyz_world'],visible=truth['visible'],identifiable=truth['identifiable'])
            radii=(320*.22/truth['xyz_world'][0,[0,5],2]).tolist()
            (folder/'setup.json').write_text(json.dumps({'radii':radii,'initial_centers':inputs['queries'][[0,5],1:].tolist()},indent=2)+'\n')
            manifest.append({'case':name,'condition':condition,'seed':seed,'split':split,'input':str((folder/'input.npz').relative_to(ROOT)),
                             'input_sha256':sha(folder/'input.npz'),'rgbd_input':str((folder/'rgbd_input.npz').relative_to(ROOT)),
                             'rgbd_sha256':sha(folder/'rgbd_input.npz'),'truth':str((folder/'truth.npz').relative_to(ROOT)),
                             'truth_sha256':sha(folder/'truth.npz'),'video_pixel_sha256':digest})
            if seed==810:
                for col,index in enumerate([0,8,16,24,32]):
                    x,y=col*384,row*276;overview.paste(Image.fromarray(inputs['video'][index]),(x,y));draw.text((x+3,y+258),f'{condition} f{index}',fill='black')
    assert len({row['video_pixel_sha256'] for row in manifest})==40
    overview.save(output/'development_overview.jpg',quality=95)
    (ROOT/'data_manifest.json').write_text(json.dumps({'cases':manifest,'source_sha256':sha(Path(__file__)),
        'frozen_geometry_source_sha256':sha(OLD/'render_controls.py'),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
        'gate_config_sha256':sha(ROOT/'gate_config.json'),'scope':'40new controlled textures/seeds; knownRGBD inputs separate from future truth.'},indent=2)+'\n')
    print(json.dumps({'cases':40,'development':10,'heldout':30,'unique_video_pixels':40,'overlap_with_prior':0},indent=2))


if __name__=='__main__':main()
