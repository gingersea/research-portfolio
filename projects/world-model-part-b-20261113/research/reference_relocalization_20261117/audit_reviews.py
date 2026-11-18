"""Run the alternative-family sampler literally; preserve defects rather than approve."""
import json
from pathlib import Path
import numpy as np
from review_matcher import ROOT

if __name__=='__main__':
    out=ROOT/'review_comparison';out.mkdir(exist_ok=False)
    cached=np.load(ROOT/'matching_audit/real_descriptors.npz',allow_pickle=False)
    raw=json.loads((ROOT/'review_protocol/response.json').read_text())['message']['content']
    review=json.loads(raw);source=review['sampler_python']
    if isinstance(source,dict):source=source['json_string']
    (out/'qwen_sampler.py').write_text(source+'\n')
    scope={'np':np}
    try:
        exec(compile(source,'qwen_sampler.py','exec'),scope)
        result=scope['sample'](cached['reference'],cached['query_xy'])
        np.testing.assert_allclose(result,cached['author_samples'],rtol=1e-4,atol=1e-5)
        status={'status':'matched'}
    except Exception as error:status={'status':'failed','error':repr(error)}
    (out/'receipt.json').write_text(json.dumps({'qwen_independent_sampler':status,'producer_author_matches_exact':True,
       'glm_independent_sampler':'failed grid dimensionality; see matching_audit/receipt.json',
       'independent_review_overall_pass':False},indent=2))
    print(json.dumps(status))
