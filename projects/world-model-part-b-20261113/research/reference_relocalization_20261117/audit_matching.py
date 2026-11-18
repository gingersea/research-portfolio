"""Compare adapter with original author sampler and per-query cosine code."""
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
    out=ROOT/'matching_audit';out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    case=json.loads((PROJECT/'research/real_measurement_20261117/manifest.json').read_text())['cases'][3]
    data=np.load(verify_file(case['files']['input']),allow_pickle=False)
    t=int(data['requested_times'][-1]);q=torch.from_numpy(data['queries_tyx'][:,[2,1]].copy())
    model,assets=load_checked('cuda:0')
    from dust3r.inference import inference
    from pips2_utils.samp import bilinear_sample2d
    with torch.inference_mode():
        result=inference([(view(data['video'][0],0),view(data['video'][t],t))],model,'cuda:0',batch_size=1,verbose=False)
        ref=result['pred1']['desc'][0].float();tar=result['pred2']['desc'][0].float()
        xy,back,accepted=descriptor_match(ref,tar,q)
        original=[]
        sample=bilinear_sample2d(ref.permute(2,0,1)[None],q[:,0][None],q[:,1][None])[0].T
        for features in sample:
            sim=F.cosine_similarity(features[None].repeat(288*512,1),tar.reshape(-1,tar.shape[-1]))
            index=(sim==sim.max()).nonzero()[0,0]
            original.append([int(index%512),int(index//512)])
        expected=np.array(original)
        np.testing.assert_array_equal(xy.numpy(),expected)
        # Original sampler also explicitly confirms the fractional edge convention.
        fractional=torch.tensor([[.75,.25],[1.75,1.5]])
        fixture=torch.eye(4).reshape(2,2,4)
        fixture_sample=bilinear_sample2d(fixture.permute(2,0,1)[None],fractional[:,0][None],fractional[:,1][None])[0].T
        np.testing.assert_allclose(fixture_sample.numpy(),[[.1875,.5625,.0625,.1875],[0,0,0,1]])
    np.savez_compressed(out/'real_descriptors.npz',reference=ref.numpy(),target=tar.numpy(),query_xy=q.numpy(),
                        xy=xy.numpy(),reverse=back.numpy(),accepted=accepted.numpy(),author_xy=expected,author_samples=sample.numpy())
    raw=json.loads((ROOT/'review_matcher/response.json').read_text())['message']['content']
    review=json.loads(raw)
    sampler=review['sampler_python']
    (out/'glm_sampler.py').write_text(sampler+'\n')
    namespace={}
    try:
        exec(compile(sampler,'glm_sampler.py','exec'),namespace)
        value=namespace['sample'](ref.numpy(),q.numpy())
        np.testing.assert_allclose(value,sample.numpy(),rtol=1e-4,atol=1e-5)
        result_b={'status':'matched'}
    except Exception as error:
        result_b={'status':'failed','error':repr(error)}
    save(out/'receipt.json',{'case':case['id'],'time':t,'queries':len(q),'input_sha256':case['files']['input']['sha256'],
        'assets':assets,'author_matches_exact':True,'fractional_author_fixture_passed':True,
        'different_family_implementation_result':result_b,'descriptor_cache_sha256':sha(out/'real_descriptors.npz'),
        'source_sha256':sha(Path(__file__)),'future_truth_read':False})
    print(json.dumps({'author_matches':len(q),'glm_implementation':result_b}))
