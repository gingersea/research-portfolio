"""Independent numerical interpretation of the development result, not new evidence."""
import hashlib
import json
import time
from review_matcher import ROOT,call

if __name__=='__main__':
    out=ROOT/'review_interpretation';out.mkdir(exist_ok=False)
    model='glm-4.7-flash:q4_K_M';metadata=next(m for m in call('/api/tags')['models'] if m['name']==model)
    files=[ROOT/'analysis_development/summary.json',ROOT/'pairing_development/summary.json',ROOT/'METHOD_NOTES.zh.md']
    prompt='''独立核对这份开发集结果的数字与解释。你是GLM，生产稿来自GPT，未提供其会话上下文。只依据原始结果JSON和待核稿。返回JSON：facts字段含native_return_coordinates_correct、descriptor_return_coordinates_correct、native_accepted_returns、descriptor_accepted_returns、new_return_coordinates、mast3r_cycle_wrong_location六个整数；alternative_conclusion_zh字段用<=120字独立写结论；unsupported_claims数组只列真实数值/逻辑矛盾，最多2条。坐标正确不等于被可见性门接受，开发集不等于新参与者验证。不要声称模型运行或整个项目通过。\n'''
    content=''.join('\nFILE '+str(p)+'\n'+p.read_text() for p in files)
    body={'model':model,'messages':[{'role':'user','content':prompt+content}],'stream':False,'think':False,'format':'json',
          'options':{'temperature':0,'num_ctx':16384,'num_predict':1300},'keep_alive':0}
    (out/'request.json').write_text(json.dumps(body,indent=2,ensure_ascii=False))
    (out/'inputs.json').write_text(json.dumps({'metadata':metadata,'producer_family':'GPT','reviewer_family':'GLM','independent_context':True,
        'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
    start=time.monotonic();response=call('/api/chat',body)
    (out/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False));(out/'REVIEW.md').write_text(response['message']['content'])
    receipt={key:response.get(key) for key in ['model','done','done_reason','prompt_eval_count','eval_count']}
    expected={'native_return_coordinates_correct':15,'descriptor_return_coordinates_correct':15,'native_accepted_returns':12,
              'descriptor_accepted_returns':15,'new_return_coordinates':0,'mast3r_cycle_wrong_location':51}
    try:
        parsed=json.loads(response['message']['content'])
        receipt['facts_match_actual_json']=parsed['facts']==expected
    except Exception as error:receipt['parse_error']=repr(error)
    receipt.update(elapsed_seconds=time.monotonic()-start,api_inference_cost_usd=0,scope='development interpretation only')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
