"""Fixed PStudio prefixes; supplied calibration, sparse targets separate."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fetch import sha

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'pilot3d';output.mkdir(exist_ok=False);rows=[]
    for row in json.loads((ROOT/'pilot3d_assets.json').read_text())['cases']:
        assert sha(row['local'])==row['sha256']
        raw=np.load(row['local'],allow_pickle=False)
        source_images=raw['images_jpeg_bytes'];T=min(64,len(source_images))
        video=np.stack([np.asarray(Image.open(io.BytesIO(bytes(s))).convert('RGB')) for s in source_images[:T]])
        q=raw['queries_xyt'];ids=np.flatnonzero(q[:,2]<T)[:8];assert len(ids)==8
        query=q[ids][:,[2,1,0]].astype(np.float32)
        fxfy=raw['fx_fy_cx_cy'];fx,fy,cx,cy=fxfy
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
        xyz=raw['tracks_XYZ'][:T,ids];vis=raw['visibility'][:T,ids]
        pixels=xyz@K.T;gtxy=pixels[...,:2]/pixels[...,2:]
        qframes=query[:,0].astype(int);query_error=np.linalg.norm(gtxy[qframes,np.arange(8)]-query[:,[2,1]],axis=-1)
        assert np.max(query_error)<1., 'Check published camera/coordinate convention before inference'
        name=Path(row['name']).stem;folder=output/name;folder.mkdir()
        np.savez_compressed(folder/'input.npz',video=video,queries_tyx=query,intrinsics=K)
        np.savez_compressed(folder/'truth.npz',tracks_xyz=xyz,tracks_xy=gtxy,visible=vis)
        record={'case':name,'source_sha256':row['sha256'],'original_frames':len(source_images),'frames':T,
                'selected_point_ids':ids.tolist(),'queries':8,'image_shape':list(video.shape[1:]),
                'input':str((folder/'input.npz').relative_to(ROOT)),'input_sha256':sha(folder/'input.npz'),
                'truth':str((folder/'truth.npz').relative_to(ROOT)),'truth_sha256':sha(folder/'truth.npz'),
                'max_query_projection_error_px':float(query_error.max()),'camera':'known static PStudio calibration',
                'annotation_type':'calibrated geometry, not manually annotated depth'}
        rows.append(record);Image.fromarray(video[0]).save(folder/'first_frame.png')
    (ROOT/'pilot3d_manifest.json').write_text(json.dumps({'cases':rows,'selection_plan_sha256':sha(ROOT/'ANALYSIS_PLAN.md'),
        'source_sha256':sha(Path(__file__))},indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
