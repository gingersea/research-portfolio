"""Different-family independent implementation and focused review, local only."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]

def call(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:11434'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response:
        return json.load(response)

if __name__=='__main__':
    out=ROOT/'review_matcher';out.mkdir(exist_ok=False)
    model='glm-4.7-flash:q4_K_M'
    metadata=next(m for m in call('/api/tags')['models'] if m['name']==model)
    source=Path('/data/AI_Research/shared/reference_relocalization_20261117/source')
    files=[PROJECT/'benchmark/physcheck/relations/descriptor.py',PROJECT/'tests/test_descriptor_relocalization.py',
           ROOT/'run_descriptor.py',ROOT/'PROTOCOL.md',source/'pips2_utils/samp.py']
    prompt='''You are GLM, independently checking GPT's reference-target descriptor matcher. Use only supplied raw code and protocol. (A) Independently calculate two fixtures: fractional q=(.75,.25) from reference rows [1,0],[0,1] against target row0 [1,0],[0,1], row1 [1,3],[-1,0]; and q=(1.75,1.5) on a2x2 one-hot identity map, with border clamping. Give forward XY. (B) Supply a small independent NumPy bilinear sampler function sample(desc,query_xy), returning NxD using border-clamped indices, to compare against producer at real descriptor inputs. Return code in JSON field sampler_python. (C) Audit code for actual discrepancies: XY versus YX, pixel centers, argmax ties, absent-point visibility claims, future-label leakage. At most3 concrete findings, no invented runs, <=70words each. Return JSON keys fixtures,sampler_python,findings,limits. Max1100words. This is not a whole study pass.\n'''
    content=''.join('\nFILE '+str(p)+'\n'+p.read_text() for p in files)
    body={'model':model,'messages':[{'role':'user','content':prompt+content}],'stream':False,'think':False,'format':'json',
          'options':{'temperature':0,'num_ctx':16384,'num_predict':2600},'keep_alive':0}
    (out/'request.json').write_text(json.dumps(body,indent=2))
    (out/'inputs.json').write_text(json.dumps({'producer_family':'GPT','reviewer_family':'GLM','endpoint':'http://127.0.0.1:11434',
         'metadata':metadata,'independent_context':True,'raw_inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
    start=time.monotonic();response=call('/api/chat',body)
    (out/'response.json').write_text(json.dumps(response,indent=2))
    (out/'REVIEW.md').write_text(response['message']['content'])
    receipt={key:response.get(key) for key in ['model','done','done_reason','prompt_eval_count','eval_count']}
    receipt.update(elapsed_seconds=time.monotonic()-start,external_inference=False,api_cost_usd=0)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
