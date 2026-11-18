"""Bounded Qwen code review; GLM design output is retained and separately ruled."""
import hashlib
import json
from pathlib import Path
import time

from review_design import ROOT,PROJECT,call


if __name__=='__main__':
    output=ROOT/'review_code'
    output.mkdir(exist_ok=False)
    model='qwen3:32b-q8_0'
    metadata=next(r for r in call('/api/tags')['models'] if r['name']==model)
    files=[PROJECT/'benchmark/physcheck/relations/sparse.py',PROJECT/'tests/test_sparse_measurement.py',
           ROOT/'run_models.py',ROOT/'prepare.py',ROOT/'analyze.py',ROOT/'PROTOCOL.md',ROOT/'PROTOCOL_v2.md']
    content=''.join('\nFILE '+str(p.relative_to(PROJECT))+'\n'+p.read_text()+'\nEND\n' for p in files)
    prompt='''You are an independent local Qwen reviewer of GPT-produced scientific Python code, with no production conversation or prior model reviews supplied. Audit the actual code and fixed protocol. First calculate the two-label fixture in test_unlabeled_zero_is_not_an_occluded_negative to check its denominator and return event. Then examine the runner/preparer/evaluator for concrete runtime errors, wrong coordinate scaling or input leakage. Unknown vis_valids=0 frames must not become negative labels. Accepted precision includes true AND false accepted known-label points. Ground-truth return events must not depend on a prediction; strict recovery is a separate metric. Reverse reciprocity is an acceptance gate, not coordinate repair. Do not infer new experimental results. Return JSON: fixture_calculation, findings (<=3 concrete findings, each with file/line/function/evidence), limits. Max500words. If no concrete defect, findings=[]. Do not claim whole scientific validation passed.\n'''
    body={'model':model,'messages':[{'role':'user','content':prompt+content}],'stream':False,'think':False,'format':'json',
          'options':{'temperature':0,'num_ctx':16384,'num_predict':2300},'keep_alive':0}
    (output/'request.json').write_text(json.dumps(body,indent=2))
    (output/'inputs.json').write_text(json.dumps({'model_metadata':metadata,'producer_family':'GPT','reviewer_family':'Qwen',
       'files':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
       'independent_context':True,'external_inference':False},indent=2))
    start=time.monotonic()
    response=call('/api/chat',body)
    (output/'response.json').write_text(json.dumps(response,indent=2))
    (output/'REVIEW.md').write_text(response['message']['content'])
    record={key:response.get(key) for key in ['model','done','done_reason','prompt_eval_count','eval_count']}
    record['elapsed_seconds']=time.monotonic()-start
    (output/'receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
