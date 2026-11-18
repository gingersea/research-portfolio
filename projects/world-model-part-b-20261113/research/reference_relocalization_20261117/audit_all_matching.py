"""Audit finite-precision global argmax against the author's literal cosine loop."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from run_descriptor import ROOT,PROJECT,load_checked,view

sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file
from physcheck.relations.descriptor import descriptor_match

if __name__=='__main__':
    out=ROOT/'matching_audit_full';out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    manifest=json.loads((PROJECT/'research/real_measurement_20261117/manifest.json').read_text())
    model,assets=load_checked('cuda:0')
    from dust3r.inference import inference
    from pips2_utils.samp import bilinear_sample2d
    records=[]
    with torch.inference_mode():
        for case in manifest['cases']:
            data=np.load(verify_file(case['files']['input']),allow_pickle=False)
            q=torch.from_numpy(data['queries_tyx'][:,[2,1]].copy())
            old=np.load(ROOT/'predictions_development'/(case['id']+'.npz'),allow_pickle=False)
            for t in data['requested_times']:
                result=inference([(view(data['video'][0],0),view(data['video'][t],int(t)))],model,'cuda:0',batch_size=1,verbose=False)
                ref=result['pred1']['desc'][0].float();tar=result['pred2']['desc'][0].float();h,w,d=ref.shape
                xy,back,accepted=descriptor_match(ref,tar,q)
                sampled=bilinear_sample2d(ref.permute(2,0,1)[None],q[:,0][None],q[:,1][None])[0].T
                expected=[]
                for features in sampled:
                    sim=F.cosine_similarity(features[None].repeat(h*w,1),tar.reshape(-1,d))
                    index=(sim==sim.max()).nonzero()[0,0]
                    expected.append([int(index%w),int(index//w)])
                expected=np.array(expected,np.float32)
                delta=np.linalg.norm(xy.numpy()-expected,axis=-1)
                row={'case':case['id'],'time':int(t),'queries':len(q),'exact_author_matches':int((delta==0).sum()),
                     'maximum_pixel_delta':float(delta.max()),'repeat_matches_previous_prediction':bool(np.array_equal(xy.numpy(),old['xy'][t]))}
                records.append(row);print(json.dumps(row),flush=True)
                np.savez_compressed(out/(case['id']+f'_{t}.npz'),query_xy=q.numpy(),adapter_xy=xy.numpy(),author_xy=expected)
    save(out/'receipt.json',{'cases':len(manifest['cases']),'pairs':len(records),'queries':sum(r['queries'] for r in records),
        'author_matches':sum(r['exact_author_matches'] for r in records),'all_repeat_predictions_exact':all(r['repeat_matches_previous_prediction'] for r in records),
        'records':records,'assets':assets,'source_sha256':sha(Path(__file__)),'future_truth_read':False,'development_only':True})
