"""Bounded follow-up after an overlong, repetitive broad GLM review."""
import json
from pathlib import Path
import time

from independent_review import request, MODEL, ROOT, PROJECT


def main():
    folder = ROOT / 'review/focused'
    folder.mkdir(exist_ok=False)
    names = ['benchmark/physcheck/relations/evaluation.py', 'tests/test_geometry_completeness.py',
             'research/proposal_completion_20261117/analysis/human_pc_probe.json',
             'research/proposal_completion_20261117/analysis/full3d.json', 'report/report.md']
    content = ''.join('\nFILE ' + n + '\n' + (PROJECT / n).read_text() + '\nEND\n' for n in names)
    prompt = '''Audit the supplied raw numerical results, scorer/tests and report. This is a bounded local independent GLM review, different family/source from GPT production and Qwen judgments. No prior review or producer explanation is supplied. Check ONLY (1) whether the report represents calibration failures as valid accuracy/ranking, (2) whether full-versus-finite 3D EPE and its missing denominator are represented faithfully, (3) arithmetic of human-PC positive=consistent confusion counts. Report up to THREE concrete discrepancies, each <=60 words with file/evidence; if no discrepancy, say so. Then one sentence of limitations. A failed experimental gate is a scientific negative result, not itself a discrepancy if clearly reported. Do not demand human labels for known procedural/geometry truth or confuse per-video values with macro means. Maximum 300 words. Return JSON with findings (array) and limits (string).\n'''
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt + content}],
            'stream': False, 'think': False, 'format': 'json',
            'options': {'temperature': 0, 'num_ctx': 16384, 'num_predict': 1600}, 'keep_alive': 0}
    (folder / 'request.json').write_text(json.dumps(body, ensure_ascii=False, indent=2))
    start = time.monotonic()
    response = request('/api/chat', body)
    (folder / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
    (folder / 'REVIEW.md').write_text(response.get('message', {}).get('content', ''))
    receipt = {key: response.get(key) for key in ['model', 'done', 'done_reason', 'prompt_eval_count', 'eval_count']}
    receipt.update(elapsed_seconds=time.monotonic()-start, independent_context=True,
                   scope='Three stated scientific consistency checks only; not full code/runtime/visual review')
    (folder / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
