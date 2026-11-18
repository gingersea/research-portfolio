"""Narrow independent Qwen numerical/interpretation review of new raw results."""
import hashlib
import json
import time
from review_matcher import ROOT,call

if __name__=='__main__':
    out=ROOT/'review_new_results';out.mkdir(exist_ok=False)
    model='qwen3:32b-q8_0';metadata=next(m for m in call('/api/tags')['models'] if m['name']==model)
    files=[ROOT/'analysis_validation/summary.json',ROOT/'pairing_validation/summary.json',ROOT/'VALIDATION_DIAGNOSTICS.zh.md']
    prompt='''独立核算新参与者实验，并审查中文诊断的逻辑。生产方GPT，你是独立Qwen，只读以下原始结果和待审稿，无历史会话。返回JSON：facts含native_located_all_visible、mast3r_located_all_visible、new_return_coordinates、lost_return_coordinates、native_accepted_returns、mast3r_accepted_returns、mast3r_cycle_wrong_location、mast3r_cycle_false_visible八个整数。坐标正确来自located_all_visible或配对记录；被接受的正确重现来自counts.recovered，不是总accepted。alternative_conclusion_zh<=100字。concrete_findings最多2条，限待审稿真实矛盾。必须区分出画与遮挡、稀疏点标签与物理违规、人群小样本与普遍结论。不要声称自己运行模型或整个项目通过。\n'''
    content=''.join('\nFILE '+str(p)+'\n'+p.read_text() for p in files)
    body={'model':model,'messages':[{'role':'user','content':prompt+content}],'stream':False,'think':False,'format':'json',
        'options':{'temperature':0,'num_ctx':16384,'num_predict':1100},'keep_alive':0}
    (out/'request.json').write_text(json.dumps(body,indent=2,ensure_ascii=False))
    (out/'inputs.json').write_text(json.dumps({'metadata':metadata,'producer_family':'GPT','reviewer_family':'Qwen',
        'independent_context':True,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
    start=time.monotonic();response=call('/api/chat',body)
    (out/'response.json').write_text(json.dumps(response,indent=2,ensure_ascii=False));(out/'REVIEW.md').write_text(response['message']['content'])
    receipt={key:response.get(key) for key in ['model','done','done_reason','prompt_eval_count','eval_count']}
    expected={'native_located_all_visible':28,'mast3r_located_all_visible':21,'new_return_coordinates':0,'lost_return_coordinates':1,
        'native_accepted_returns':6,'mast3r_accepted_returns':7,'mast3r_cycle_wrong_location':19,'mast3r_cycle_false_visible':1}
    try:receipt['facts_match_actual_json']=json.loads(response['message']['content'])['facts']==expected
    except Exception as error:receipt['parse_error']=repr(error)
    receipt.update(elapsed_seconds=time.monotonic()-start,api_inference_cost_usd=0,scope='new-results numerical/interpretation review only')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
