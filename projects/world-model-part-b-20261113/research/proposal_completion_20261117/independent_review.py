"""Isolated, local, different-family review from raw materials, not conversation."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MODEL = 'glm-4.7-flash:q4_K_M'


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('http://127.0.0.1:11434' + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def main():
    folder = ROOT / 'review'
    folder.mkdir(exist_ok=False)
    tags = request('/api/tags')
    metadata = next(x for x in tags['models'] if x['name'] == MODEL)
    inputs = [
        ROOT / 'PROTOCOL.md', ROOT / 'analysis/full3d.json', ROOT / 'analysis/human_pc_probe.json',
        ROOT / 'analysis/gravity_judge_gate.json', ROOT / 'analysis/davis_replication.json',
        PROJECT / 'report/report.md', PROJECT / 'benchmark/physcheck/relations/evaluation.py',
        PROJECT / 'benchmark/physcheck/relations/backends.py', PROJECT / 'benchmark/physcheck/relations/pipeline.py',
        PROJECT / 'benchmark/physcheck/judge/local_vlm.py', ROOT / 'analyze_new.py',
        PROJECT / 'tests/test_geometry_completeness.py', PROJECT / 'tests/test_relations_public.py',
        ROOT / 'prepare_judges.py', ROOT / 'run_visual_judge.py', ROOT / 'GEOMETRY_CORRECTION.md']
    texts = []
    hashes = {}
    for path in inputs:
        raw = path.read_bytes()
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(raw).hexdigest()
        texts.append(f'\nFILE {path.relative_to(PROJECT)}\n{raw.decode()}\nEND FILE\n')
    # No production conversation or arguments justifying conclusions are supplied.
    prompt = '''Independently audit this Python physical-video research delivery. You are the final local GLM-family reviewer, distinct from GPT code/report production and Qwen visual judgments. Read the provided raw protocol, saved numerical results, code and draft report. Identify concrete scientific/code errors, truth leakage, denominator omissions, claimed acceptance despite failed gates, unfair comparisons, misleading claims, or important untested paths. Calibrations must pass all controls before scores are valid. Frames/queries are not independent videos. PStudio is the same 3 previously used videos with longer inference, not unseen validation. VideoPhy2 positive class means consistent. Current AGENTS forbids accessing gingersea, publication and Google; local package is permitted. No new tuning is authorized. Judge opinions are not human ground truth. Do not infer runtime or visual correctness solely from metadata. The frozen protocol's reviewer-family sentence predates the choice to use Qwen as judge; this final reviewer is GLM as required by the different-source/family rule. Return concise numbered findings with severity, exact file/function and evidence. Separate blockers from optional limitations; explicitly state what supplied material cannot establish. If there are no concrete blockers, say so, but do not declare the scientific goals passed. Do not rewrite the report.\n'''
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt + ''.join(texts)}],
               'stream': False, 'think': False, 'options': {'temperature': 0, 'num_ctx': 32768, 'num_predict': 4096}, 'keep_alive': 0}
    (folder / 'request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (folder / 'input_manifest.json').write_text(json.dumps({'model_metadata': metadata, 'inputs': hashes,
         'input_characters': len(payload['messages'][0]['content']), 'independent_context': True,
         'producer_family': 'GPT', 'visual_judge_family': 'Qwen', 'reviewer_family': 'GLM',
         'external_inference': False, 'endpoint': 'http://127.0.0.1:11434/api/chat'}, indent=2))
    started = time.monotonic()
    response = request('/api/chat', payload)
    (folder / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
    (folder / 'REVIEW.md').write_text(response.get('message', {}).get('content', ''))
    receipt = {'model': response.get('model'), 'done': response.get('done'), 'done_reason': response.get('done_reason'),
               'prompt_eval_count': response.get('prompt_eval_count'), 'eval_count': response.get('eval_count'),
               'elapsed_seconds': time.monotonic() - started, 'request_sha256': hashlib.sha256((folder / 'request.json').read_bytes()).hexdigest()}
    (folder / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
