"""Export only this calibration's session, access, image and judgment hashes.

Never export conversations, hidden reasoning, credentials or full rollouts.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

STUDY = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-log', type=Path, required=True)
    parser.add_argument('--child-log-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite first session evidence')
    manifest = {row['case_id']: row for row in json.loads((STUDY / 'judge_input_manifest.json').read_text())}
    with args.parent_log.open() as stream:
        parent = json.loads(next(stream))['payload']['id']
    spawns = {}
    for line in args.parent_log.open():
        event = json.loads(line)
        payload = event.get('payload', {})
        if event['type'] != 'response_item' or payload.get('type') != 'function_call' or payload.get('name') != 'spawn_agent':
            continue
        invocation = json.loads(payload['arguments'])
        name = invocation.get('task_name', '')
        if not name.startswith('pgjudge_c_'):
            continue
        identifier = name.removeprefix('pgjudge_')
        assert identifier in manifest and identifier not in spawns
        spawns[identifier] = {'fork_turns': invocation.get('fork_turns'), 'requested_role': invocation.get('agent_type'), 'spawn_call_id': payload['call_id'], 'spawn_arguments_sha256': sha(payload['arguments'].encode()), 'spawn_timestamp': event['timestamp']}
    receipts = {}
    for path in args.child_log_dir.glob('rollout-*.jsonl'):
        with path.open() as stream:
            first = json.loads(next(stream))
        if first['type'] != 'session_meta':
            continue
        meta = first['payload']
        source = meta.get('source', {})
        if not isinstance(source, dict):
            continue
        spawned = source.get('subagent', {}).get('thread_spawn', {})
        agent_path = spawned.get('agent_path', '')
        if spawned.get('parent_thread_id') != parent or not agent_path.startswith('/root/pgjudge_c_'):
            continue
        identifier = agent_path.removeprefix('/root/pgjudge_')
        assert identifier in manifest and identifier not in receipts
        events = [json.loads(line) for line in path.open()]
        models = sorted({event['payload']['model'] for event in events if event['type'] == 'turn_context'})
        efforts = sorted({event['payload'].get('effort') for event in events if event['type'] == 'turn_context'})
        allowed = Path(manifest[identifier]['public_input_dir'])
        judgment_path = STUDY / 'judgments' / f'{identifier}.json'
        hashes, accesses, tool_records = [], set(), []
        for event in events:
            payload = event.get('payload', {})
            if event['type'] != 'response_item':
                continue
            if payload.get('type') == 'custom_tool_call':
                content = payload.get('input', '')
                # Only file paths in actual tool inputs are audited. Generic
                # inherited instructions and conversation text are not read/exported.
                for candidate in re.findall(r'/(?:data|home|tmp)/[^\s\"\'\\)]+', content):
                    candidate = candidate.rstrip('},;]')
                    observed = Path(candidate)
                    if observed != allowed and not observed.is_relative_to(allowed) and observed != judgment_path:
                        accesses.add(candidate)
                tool_records.append({'call_id': payload['call_id'], 'tool_input_sha256': sha(content.encode())})
            if payload.get('type') == 'custom_tool_call_output' and isinstance(payload.get('output'), list):
                for item in payload['output']:
                    if isinstance(item, dict) and item.get('type') == 'input_image':
                        url = item.get('image_url', '')
                        if isinstance(url, str) and url.startswith('data:') and ';base64,' in url:
                            hashes.append(sha(base64.b64decode(url.split(',', 1)[1])))
        case_payload = json.loads((allowed / 'payload.json').read_text())
        expected_hashes = [row['sha256'] for row in case_payload['sheets']]
        assert sha((allowed / 'payload.json').read_bytes()) == manifest[identifier]['payload_sha256']
        assert sha((allowed / 'rubric.md').read_bytes()) == manifest[identifier]['rubric_sha256']
        actual_spawn = spawns[identifier]
        receipts[identifier] = {'session_id': meta['id'], 'parent_session_id': parent, 'agent_path': agent_path, 'actual_role': spawned.get('agent_role'), 'models_observed': models, 'efforts_observed': efforts, 'model_provider': meta.get('model_provider'), **actual_spawn, 'fork_none_verified': actual_spawn['fork_turns'] == 'none', 'actual_image_sha256': hashes, 'expected_image_sha256': expected_hashes, 'all_assigned_images_viewed': all(value in hashes for value in expected_hashes), 'outside_input_paths': sorted(accesses), 'private_or_outside_input_access': bool(accesses), 'judgment_sha256': sha(judgment_path.read_bytes()), 'input_payload_sha256': manifest[identifier]['payload_sha256'], 'tool_records': tool_records}
    assert set(receipts) == set(manifest)
    assert len({row['session_id'] for row in receipts.values()}) == 12
    assert all(row['models_observed'] == ['gpt-6-astra'] and row['efforts_observed'] == ['high'] and row['actual_role'] == 'lazycodex-qa-executor' for row in receipts.values())
    result = {'schema': 'filtered-principia-session-evidence-v1', 'receipts': receipts, 'model_version_limit': 'Actual turn metadata exposes a model alias, not a provider-dated snapshot.', 'read_scope_limit': 'Audits literal paths in exposed tool calls and actual image hashes; fresh context and scoped instructions are not an OS filesystem sandbox. Generic unexposed/runtime behavior is not certified.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_sessions': len(receipts), 'all_images_verified': all(row['all_assigned_images_viewed'] for row in receipts.values()), 'any_outside_access': any(row['private_or_outside_input_access'] for row in receipts.values())}))


if __name__ == '__main__':
    main()
