"""Export only this experiment's verifiable session/tool metadata.

Never export conversation text, hidden reasoning, credentials, encrypted prompts
or full transcripts. Models, unique child IDs, fork mode and image output events
are read from the real local rollout records rather than inferred from filenames.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

STUDY=Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-log',type=Path,required=True)
    parser.add_argument('--child-log-dir',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--output',type=Path,default=STUDY/'run/judge_session_receipts.json')
    args=parser.parse_args()
    with args.parent_log.open() as stream:
        parent=json.loads(next(stream))['payload']['id']
    spawns={}
    for line in args.parent_log.open():
        event=json.loads(line);payload=event.get('payload',{})
        if event['type']!='response_item' or payload.get('type')!='function_call' or payload.get('name')!='spawn_agent':
            continue
        invocation=json.loads(payload['arguments'])
        if invocation.get('agent_type')!='lazycodex-qa-executor' or not invocation.get('task_name','').startswith('judge_v_'):
            continue
        name=invocation['task_name']
        if name in spawns:
            raise ValueError(f'duplicate registered judge invocation: {name}')
        spawns[name]={'fork_turns':invocation.get('fork_turns'),'spawn_timestamp':event['timestamp'],
                      'spawn_call_id':payload['call_id'],'spawn_arguments_sha256':sha(payload['arguments'].encode())}
    completed={p.stem for p in (STUDY/'run/judgments').glob('*.json')}
    receipts={}
    for path in sorted(args.child_log_dir.glob('rollout-*.jsonl')):
        with path.open() as stream:
            first=json.loads(next(stream))
        if first['type']!='session_meta':continue
        meta=first['payload']
        source=meta.get('source',{})
        if not isinstance(source,dict):continue
        spawn=source.get('subagent',{}).get('thread_spawn',{})
        agent_path=spawn.get('agent_path','')
        if spawn.get('parent_thread_id')!=parent or not agent_path.startswith('/root/judge_v_'):
            continue
        vid=agent_path.removeprefix('/root/judge_')
        if vid not in completed:continue
        if vid in receipts:raise ValueError(f'duplicate judge child thread: {vid}')
        events=[json.loads(line) for line in path.open()]
        models=sorted({e['payload']['model'] for e in events if e['type']=='turn_context'})
        efforts=sorted({e['payload'].get('effort') for e in events if e['type']=='turn_context'})
        image_calls=[];image_outputs=0;image_hashes=[]
        for event in events:
            payload=event.get('payload',{})
            if event['type']!='response_item':continue
            if payload.get('type')=='custom_tool_call':
                assert str(STUDY/'run/private') not in payload.get('input',''),vid
            if payload.get('type')=='custom_tool_call' and 'view_image' in payload.get('input',''):
                image_calls.append({'timestamp':event['timestamp'],'call_id':payload['call_id'],
                                    'tool_input_sha256':sha(payload['input'].encode())})
            if payload.get('type')=='custom_tool_call_output' and isinstance(payload.get('output'),list):
                image_outputs+=sum(item.get('type')=='input_image' for item in payload['output'] if isinstance(item,dict))
                for item in payload['output']:
                    if isinstance(item,dict) and item.get('type')=='input_image':
                        url=item.get('image_url','')
                        if isinstance(url,str) and url.startswith('data:') and ';base64,' in url:
                            image_hashes.append(sha(base64.b64decode(url.split(',',1)[1])))
        actual=spawns['judge_'+vid]
        assert actual['fork_turns']=='none'
        assert models==['gpt-6-astra'] and efforts==['high']
        assert spawn.get('agent_role')=='lazycodex-qa-executor'
        assert image_calls and image_outputs>0,vid
        base=STUDY/'run/blind/frames'/vid
        assert sha((base/'grid.png').read_bytes()) in image_hashes,vid
        receipts[vid]={'session_id':meta['id'],'parent_session_id':parent,'agent_path':agent_path,
                        'agent_role':spawn['agent_role'],'model':models[0],'effort':efforts[0],
                        'model_provider':meta.get('model_provider'),**actual,
                        'image_calls':image_calls,'image_output_events':image_outputs,
                        'actual_image_input_sha256':image_hashes,'archived_grid_was_actually_viewed':True,
                        'grid_sha256':sha((base/'grid.png').read_bytes()),
                        'payload_sha256':sha((base/'payload.json').read_bytes()),
                        'judgment_sha256':sha((STUDY/'run/judgments'/f'{vid}.json').read_bytes())}
    assert set(receipts)==completed,(completed-set(receipts))
    if not args.partial:assert len(receipts)==56
    assert len({r['session_id'] for r in receipts.values()})==len(receipts)
    result={'schema':'filtered-session-evidence-v1','complete':not args.partial,'receipts':receipts,
            'excluded':'conversation text, reasoning, credentials, encrypted prompt values and complete transcripts',
            'model_version_limit':'actual turn metadata exposes model alias, not a dated provider snapshot'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Actual independent sessions and image events verified: {len(receipts)}')


if __name__=='__main__':main()
