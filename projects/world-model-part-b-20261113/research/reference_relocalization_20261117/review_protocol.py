"""Independent Qwen audit before new participant predictions."""
import hashlib
import json
import time
from pathlib import Path
from review_matcher import ROOT,PROJECT,call

if __name__=='__main__':
    out=ROOT/'review_protocol';out.mkdir(exist_ok=False)
    model='qwen3:32b-q8_0';metadata=next(m for m in call('/api/tags')['models'] if m['name']==model)
    files=[ROOT/'PROTOCOL.md',ROOT/'prepare_validation.py',ROOT/'run_native.py',ROOT/'run_descriptor.py',ROOT/'analyze.py',
           PROJECT/'benchmark/physcheck/relations/descriptor.py',PROJECT/'benchmark/physcheck/relations/sparse.py']
    prompt='''Independently audit this real-video correspondence experiment. GPT is producer, you are Qwen with separate context and raw sources. Before any new validation outcomes: verify participant-disjoint selection is label-stratified (OOV-return enrichment), not representative of all videos; no future coordinate/visibility labels enter inference except initial queries and requested schedule; all sparse unknown labels excluded and all failures retained; reciprocity is not identity or visibility truth; MASt3R is public static baseline, not PointSt3R weights. Check for concrete runtime/input/mask/coordinate bugs in supplied code. Return JSON concrete_findings (up to3, quote actual function/evidence), limitations (up to3). Also independently implement a NumPy sampler sample(desc,query_xy) for HxWxD, Nx2 pixel XY; floor indices then clamp four pixel indices to border, use fractional x-floor(x),y-floor(y) weights; return NxD. Do not use torch or normalized grids in this independent sampler. Include sampler_python JSON string. Max1000words. No invented runs or whole-study approval.\n'''
    content=''.join('\nFILE '+str(p)+'\n'+p.read_text() for p in files)
    body={'model':model,'messages':[{'role':'user','content':prompt+content}],'stream':False,'think':False,'format':'json',
        'options':{'temperature':0,'num_ctx':16384,'num_predict':2600},'keep_alive':0}
    (out/'request.json').write_text(json.dumps(body,indent=2))
    (out/'inputs.json').write_text(json.dumps({'metadata':metadata,'producer_family':'GPT','reviewer_family':'Qwen','independent_context':True,
        'raw_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
    start=time.monotonic();response=call('/api/chat',body)
    (out/'response.json').write_text(json.dumps(response,indent=2));(out/'REVIEW.md').write_text(response['message']['content'])
    receipt={key:response.get(key) for key in ['model','done','done_reason','prompt_eval_count','eval_count']}
    receipt.update(elapsed_seconds=time.monotonic()-start,external_inference=False,api_cost_usd=0)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
