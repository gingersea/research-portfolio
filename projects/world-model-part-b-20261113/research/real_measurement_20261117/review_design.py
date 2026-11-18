"""Independent local GLM review of sparse-mask scoring before new results."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]


def call(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:11434'+path,
       data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as response:
        return json.load(response)


if __name__=='__main__':
    output=ROOT/'review_design'
    output.mkdir(exist_ok=False)
    model='glm-4.7-flash:q4_K_M'
    metadata=next(r for r in call('/api/tags')['models'] if r['name']==model)
    inputs=[PROJECT/'benchmark/physcheck/relations/sparse.py',PROJECT/'tests/test_sparse_measurement.py',
            ROOT/'PROTOCOL.md',ROOT/'PROTOCOL_v2.md',ROOT/'sources/README.md']
    text=''.join('\nFILE '+str(p.relative_to(PROJECT))+'\n'+p.read_text()+'\nEND\n' for p in inputs)
    prompt='''Independently audit this sparse-video-label measurement scorer and protocol. Producer is GPT; you are local GLM with separate context and raw materials. Check specifically whether unknown visibility frames can enter the false-positive denominator; whether accepted known-label failures are omitted from precision; whether OOV-to-return scoring accidentally uses unannotated frames; whether reciprocity is wrongly claimed to repair coordinates. Return JSON with at most3 concrete findings, each <=70words, plus limits. If no concrete discrepancy, findings=[]. Do not treat declared scientific limitations as implementation errors. Do not invent unseen runtime or demand human labels instead of existing expert annotations. Maximum300words.\n'''
    body={'model':model,'messages':[{'role':'user','content':prompt+text}],'stream':False,'think':False,'format':'json',
          'options':{'temperature':0,'num_ctx':16384,'num_predict':1400},'keep_alive':0}
    (output/'request.json').write_text(json.dumps(body,indent=2))
    (output/'inputs.json').write_text(json.dumps({'model_metadata':metadata,'producer_family':'GPT','reviewer_family':'GLM',
        'raw_files':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'independent_context':True,'external_inference':False},indent=2))
    start=time.monotonic()
    response=call('/api/chat',body)
    (output/'response.json').write_text(json.dumps(response,indent=2))
    (output/'REVIEW.md').write_text(response['message']['content'])
    record={key:response.get(key) for key in ['model','done','done_reason','prompt_eval_count','eval_count']}
    record['elapsed_seconds']=time.monotonic()-start
    (output/'receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
