"""Frozen, label-isolated MASt3R descriptor inference at requested timestamps."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file
from physcheck.relations.descriptor import descriptor_match


def view(rgb,index):
    h,w=rgb.shape[:2]
    tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    return {'img':(tensor-.5)[None]/.5,'true_shape':np.int32([[h,w]]),'idx':index,
            'instance':f'frame{index}','mask':~(tensor.sum(0)[None]<=.01),
            'dynamic_mask':torch.zeros((1,h,w),dtype=torch.bool)}


def load_checked(device):
    assets=json.loads((ROOT/'implementation_assets.json').read_text())
    source=Path(assets['source_directory'])
    assert all(sha(source/r['file'])==r['sha256'] for r in assets['files'])
    weights=json.loads((ROOT/'checkpoint.json').read_text())
    assert sha(Path(weights['checkpoint']))==weights['sha256']
    sys.path.insert(0,str(source))
    from mast3r.model import AsymmetricMASt3R
    original=AsymmetricMASt3R.load_state_dict
    loading=[]
    def checked(self,state_dict,strict=True,**kwargs):
        result=original(self,state_dict,strict=strict,**kwargs)
        loading.append({'missing_keys':list(result.missing_keys),'unexpected_keys':list(result.unexpected_keys)})
        assert not result.missing_keys and not result.unexpected_keys,result
        return result
    AsymmetricMASt3R.load_state_dict=checked
    try:
        model=AsymmetricMASt3R.from_pretrained(weights['checkpoint']).to(device).eval()
    finally:
        AsymmetricMASt3R.load_state_dict=original
    assert len(loading)==1
    return model,{'checkpoint_sha256':weights['sha256'],'source_revision':assets['revision'],
                  'state_loading':loading,'source_manifest_sha256':sha(ROOT/'implementation_assets.json')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--freeze',type=Path)
    args=parser.parse_args()
    if args.freeze:
        freeze=json.loads(args.freeze.read_text())
        assert freeze['manifest_sha256']==sha(args.manifest)
        for file,digest in freeze['sources'].items(): assert sha(Path(file))==digest
    manifest=json.loads(args.manifest.read_text())
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261117)
    torch.cuda.set_device(args.gpu)
    device=f'cuda:{args.gpu}'
    model,assets=load_checked(device)
    from dust3r.inference import inference
    save(args.output/'runtime.json',{'utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
         'numpy':np.__version__,'python':sys.version,'assets':assets,'dtype':'float32',
         'rope':'released PyTorch fallback; no optional CUDA extension built',
         'image_storage':'512x288 RGB from existing OpenCV INTER_LINEAR preparation; no new resize',
         'pointst3r_finetuning_or_visibility_head':False})
    for case in manifest['cases']:
        started=time.monotonic()
        record={'case':case['id'],'input_sha256':case['files']['input']['sha256'],
                'manifest_sha256':sha(args.manifest),'future_truth_read':False,'requested_timestamps_provided':True,
                'runner_sha256':sha(Path(__file__)),'matcher_sha256':sha(PROJECT/'benchmark/physcheck/relations/descriptor.py'),
                'model':'public MASt3R static baseline, not PointSt3R fine-tuned','gpu':args.gpu,**assets}
        print(json.dumps({'event':'start','case':case['id']}),flush=True)
        try:
            data=np.load(verify_file(case['files']['input']),allow_pickle=False)
            rgb=data['video']; queries=data['queries_tyx']; requested=data['requested_times']
            t,h,w=rgb.shape[:3]; n=len(queries)
            assert (h,w)==(288,512) and (queries[:,0]==0).all()
            xy=np.full((t,n,2),np.nan,np.float32)
            raw=np.zeros((t,n),bool); accepted=np.zeros((t,n),bool)
            reverse=np.full((len(requested),n,2),np.nan,np.float32)
            with torch.inference_mode():
                for slot,time_index in enumerate(requested):
                    torch.cuda.empty_cache()
                    result=inference([(view(rgb[0],0),view(rgb[time_index],int(time_index)))],model,device,batch_size=1,verbose=False)
                    desc1=result['pred1']['desc'][0].float()
                    desc2=result['pred2']['desc'][0].float()
                    assert tuple(desc1.shape[:2])==(h,w) and desc1.shape==desc2.shape
                    point,back,cycle=descriptor_match(desc1,desc2,torch.from_numpy(queries[:,[2,1]].copy()))
                    xy[time_index]=point.cpu().numpy(); raw[time_index]=np.isfinite(xy[time_index]).all(-1)
                    accepted[time_index]=cycle.cpu().numpy(); reverse[slot]=back.cpu().numpy()
                    print(json.dumps({'event':'pair_done','case':case['id'],'time':int(time_index),'queries':n}),flush=True)
            path=args.output/(case['id']+'.npz')
            np.savez_compressed(path,xy=xy,raw=raw,reciprocal=accepted,reverse_xy=reverse,requested_times=requested)
            record.update(status='ok',prediction_sha256=sha(path),frames=t,queries=n)
        except Exception as error:
            record.update(status='inference_failed',error=repr(error),traceback=traceback.format_exc(),planned_case_retained=True)
            torch.cuda.empty_cache()
        record['elapsed_seconds']=time.monotonic()-started
        save(args.output/(case['id']+'.json'),record)
        print(json.dumps({'event':'done','case':case['id'],'status':record['status'],'seconds':record['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
