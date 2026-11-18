"""Official first queries with human targets separated from model inputs."""
import hashlib
import json
from pathlib import Path
import pickle

import mediapy as media
import numpy as np
from PIL import Image,ImageDraw

from fetch import sha
from official_numpy import sample_queries_first

ROOT=Path(__file__).resolve().parent


def main():
    assets=json.loads((ROOT/'dataset_assets.json').read_text())
    candidates=[r for r in assets['files'] if r['file'].endswith('.pkl')]
    assert len(candidates)==1
    path=Path(assets['release_directory'])/candidates[0]['file']
    assert sha(path)==candidates[0]['sha256']
    assert assets['md5']=='a7e786a1a5d26ba941b7c10bed21c20e'
    with path.open('rb') as f:dataset=pickle.load(f)
    assert isinstance(dataset,dict) and len(dataset)==30
    output=ROOT/'data';output.mkdir(exist_ok=False)
    rows=[];pixels=set();sheet=Image.new('RGB',(6*256,5*280),'white');draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(sorted(dataset)):
        example=dataset[name];original=example['video']
        assert original.ndim==4 and original.shape[-1]==3 and original.dtype==np.uint8
        video=original if original.shape[1:3]==(256,256) else media.resize_video(original,(256,256))
        assert video.dtype==np.uint8 and video.shape[1:]==(256,256,3)
        tracks=np.array(example['points'],dtype=np.float64,copy=True)*np.array([256,256])
        occluded=np.array(example['occluded'],dtype=bool,copy=True)
        converted=sample_queries_first(occluded,tracks,video)
        query=converted['query_points'][0].astype(np.float32)
        truth=converted['target_points'][0];occ=converted['occluded'][0]
        assert truth.shape[0]==len(query) and truth.shape[1]==len(video)
        folder=output/name;folder.mkdir()
        np.savez_compressed(folder/'input.npz',video=video,queries_tyx=query)
        np.savez_compressed(folder/'truth.npz',target_points=truth,occluded=occ)
        digest=hashlib.sha256(video.tobytes()).hexdigest();assert digest not in pixels;pixels.add(digest)
        rows.append({'case':name,'index':i,'frames':len(video),'queries':len(query),'original_shape':list(original.shape),
                     'input':str((folder/'input.npz').relative_to(ROOT)),'input_sha256':sha(folder/'input.npz'),
                     'truth':str((folder/'truth.npz').relative_to(ROOT)),'truth_sha256':sha(folder/'truth.npz'),
                     'video_pixel_sha256':digest,'query_mode':'first','coordinate_scale_xy':[256,256]})
        x,y=(i%6)*256,(i//6)*280;sheet.paste(Image.fromarray(video[0]),(x,y));draw.text((x+5,y+258),name,fill='black')
        print(json.dumps({'case':name,'frames':len(video),'queries':len(query)}),flush=True)
    sheet.save(output/'input_overview.jpg',quality=95)
    record={'cases':rows,'official_release_sha256':assets['sha256'],'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
            'source_sha256':sha(Path(__file__)),'resize_dependency':'mediapy1.2.7; identical API as official reader',
            'query_function_sha256':sha(ROOT/'official_numpy.py'),'human_annotations':True,
            'future_truth_separated':True,'total_frames':sum(r['frames'] for r in rows),'total_queries':sum(r['queries'] for r in rows)}
    (ROOT/'data_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
