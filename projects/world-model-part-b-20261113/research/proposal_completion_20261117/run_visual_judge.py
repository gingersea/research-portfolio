"""First local visual replies. One fresh prompt, no reference labels per call."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.judge.local_vlm import LocalVisualJudge, parse_pc
from physcheck.relations.contracts import save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', choices=['gravity', 'pc'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    assets = json.loads((ROOT / 'vlm_assets.json').read_text())
    directory = Path(assets['directory'])
    for file in assets['files']:
        assert sha(directory / file['file']) == file['sha256']
    model = LocalVisualJudge(str(directory), args.gpu)
    cases = json.loads((ROOT / f'{args.cases}_judge_cases.json').read_text())
    cases.sort(key=lambda c: c['split'] not in ['calibration'])
    output = ROOT / f'{args.cases}_judgments'
    output.mkdir(exist_ok=False)
    for case in cases:
        source = Path(case['input'])
        payload = json.loads(source.read_text())
        record = {'id': case['id'], 'input_sha256': sha(source), 'split': case['split'],
                  'weights_revision': assets['revision'], 'labels_loaded_by_judge': False}
        print(json.dumps({'event': 'start', 'case': case['id'], 'kind': args.cases}), flush=True)
        try:
            if payload.get('unavailable'):
                raise ValueError('Video unavailable; original planned slot retained')
            images = []
            for image in payload['images']:
                path = source.parent / image['path']
                assert sha(path) == image['sha256']
                images.append(Image.open(path).convert('RGB'))
            text, runtime = model.generate(images, payload['prompt'])
            (output / (case['id'] + '.txt')).write_text(text)
            record.update(runtime=runtime, viewed_image_sha256=[im['sha256'] for im in payload['images']])
            if args.cases == 'pc':
                parsed = parse_pc(text, case['id'], payload['frame_indices'])
            else:
                start = text.find('{')
                if start < 0:
                    raise ValueError('No structured gravity reply')
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])
                # The original frozen gravity schema validator reads no labels.
                sys.path.insert(0, str(PROJECT / 'research/principia_gravity_20261114'))
                from judge_gate import validate
                validate(parsed, {'case_id': case['id'], 'frame_count': payload['frame_count'], 'sheets': payload['sheets']})
            record.update(status='ok', judgment=parsed)
        except Exception as error:
            record.update(status='judge_failed', error=repr(error), planned_slot_retained=True)
        save(output / (case['id'] + '.json'), record)
        print(json.dumps({'event': 'done', 'case': case['id'], 'status': record['status']}), flush=True)
    save(output / 'execution_receipt.json', {'case_count': len(cases), 'first_replies_retained': True,
         'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'runner_sha256': sha(Path(__file__)),
         'local_vlm_module_sha256': sha(PROJECT / 'benchmark/physcheck/judge/local_vlm.py'),
         'fresh_context_per_case': True, 'other_case_messages_or_reference_labels_provided': False,
         'rejudging_to_pass': False, 'external_inference': False})


if __name__ == '__main__':
    main()
