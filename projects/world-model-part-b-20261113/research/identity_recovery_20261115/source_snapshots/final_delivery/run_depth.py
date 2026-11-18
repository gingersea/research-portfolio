"""Official estimated depth on controlled RGB and all four real I2V clips."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from fetch_depth import sha

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--output',default='estimated_depth_native_attention');args=parser.parse_args()
    assets=json.loads((ROOT/'depth_assets.json').read_text())
    source=Path(assets['source_directory'])
    assert all(sha(source/r['file'])==r['sha256'] for r in assets['files'])
    assert sha(assets['checkpoint'])==assets['checkpoint_sha256']
    sys.path.insert(0,str(source/'metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.dinov2_layers import attention
    # Select the author's existing FP32 attention fallback; no source/weights change.
    attention.XFORMERS_AVAILABLE=False
    torch.set_num_threads(4);torch.cuda.set_device(args.gpu);torch.manual_seed(20261115)
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    model.load_state_dict(torch.load(assets['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model=model.to(f'cuda:{args.gpu}').eval()
    cases=[]
    for row in json.loads((ROOT/'data_manifest.json').read_text())['cases']:
        if row['split']=='heldout' and row['condition'] in ['clean','occlusion','camera_motion','identity_replacement']:
            cases.append({'case':row['case'],'domain':'control','input':str(ROOT/row['input']),'input_sha256':row['input_sha256']})
    for row in json.loads((ROOT/'real_manifest.json').read_text()):
        cases.append({'case':row['clip'],'domain':'real_i2v','input':row['source'],'input_sha256':row['source_sha256']})
    output=ROOT/args.output;output.mkdir(exist_ok=False)
    for case in cases:
        assert sha(case['input'])==case['input_sha256']
        inputs=np.load(case['input']);rgb=inputs['video'] if 'video' in inputs.files else inputs[inputs.files[0]]
        start=time.monotonic();pred=[]
        print(json.dumps({'event':'start',**case}),flush=True)
        try:
            with torch.inference_mode():
                for image in rgb:pred.append(model.infer_image(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),input_size=518))
            depth=np.stack(pred).astype(np.float32)
            assert depth.shape==rgb.shape[:3] and np.isfinite(depth).all() and (depth>0).all()
            path=output/f'{case["case"]}.npz';np.savez_compressed(path,depth=depth)
            receipt={**case,'status':'ok','depth_sha256':sha(path),'source_revision':assets['source_revision'],
                     'checkpoint_sha256':assets['checkpoint_sha256'],'dtype':'float32','input_size':518,'max_depth':20,
                     'frames':len(rgb),'future_truth_read':False,'runner_sha256':sha(Path(__file__)),
                     'attention_backend':'author_native_pytorch','author_source_modified':False,
                     'elapsed_seconds':time.monotonic()-start}
        except Exception as error:
            receipt={**case,'status':'api_failure','exception':repr(error)}
            (output/f'{case["case"]}.json').write_text(json.dumps(receipt,indent=2)+'\n');raise
        (output/f'{case["case"]}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'event':'complete',**receipt}),flush=True)


if __name__=='__main__':main()
