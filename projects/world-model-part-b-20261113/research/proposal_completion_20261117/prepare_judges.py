"""Export model-only visual inputs, labels/mappings retained outside each payload."""
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha

PC_TEMPLATE = '''Evaluate visible physical commonsense in the supplied chronological sampled video frames. You are not given the original generation prompt. Ignore visual polish and do not guess the generator. Score physical plausibility from1 (clearly physically inconsistent) to5 (clearly physically plausible);3 expresses uncertainty. Describe only visible evidence and do not invent unseen motion between samples. Return JSON only with video_id, pc_score (integer1..5), reason, violations (list of {frame_idx: original printed frame index, category: gravity/support/collision/fluid/rigid_body/other, description}). An empty violations list is allowed when no specific visible violation can be established. Assigned video_id: {id}.'''


def main():
    gravity = PROJECT / 'research/principia_gravity_20261114'
    rubric = (gravity / 'JUDGE_RUBRIC.md').read_text()
    cases = []
    index = json.loads((gravity / 'judge_input_manifest.json').read_text())
    mapping = []
    for n, row in enumerate(index):
        case_id = f'g_{n:03d}'
        source = Path(row['public_input_dir'])
        payload = json.loads((source / 'payload.json').read_text())
        folder = ROOT / 'judge_inputs' / case_id
        folder.mkdir(parents=True, exist_ok=False)
        images = []
        for sheet in payload['sheets']:
            shutil.copy2(source / sheet['path'], folder / sheet['path'])
            assert sha(folder / sheet['path']) == sheet['sha256']
            images.append({'path': sheet['path'], 'sha256': sheet['sha256']})
        prompt = rubric + f'\nAssigned case_id: {case_id}. Actual frame_count: {payload["frame_count"]}. Actual sheets: {len(images)}.\n'
        save(folder / 'input.json', {'id': case_id, 'kind': 'gravity', 'images': images,
              'prompt': prompt, 'frame_count': payload['frame_count'], 'sheets': payload['sheets']})
        cases.append({'id': case_id, 'kind': 'gravity', 'split': 'calibration', 'input': str(folder / 'input.json')})
        mapping.append({'id': case_id, 'original_case_id': row['case_id']})
    prior = PROJECT / 'research/integrated_framework_20261115/fresh_i2v4_final'
    for n in range(1, 5):
        case_id = f'i_{n:03d}'
        source = prior / f'i2v_C{n}' / 'blind_visual_input'
        payload = json.loads((source / 'visual_judge_payload.json').read_text())
        folder = ROOT / 'judge_inputs' / case_id
        folder.mkdir(parents=True, exist_ok=False)
        images = []
        for sheet in payload['sheets']:
            shutil.copy2(source / sheet['path'], folder / sheet['path'])
            assert sha(folder / sheet['path']) == sheet['sha256']
            images.append({'path': sheet['path'], 'sha256': sheet['sha256']})
        save(folder / 'input.json', {'id': case_id, 'kind': 'gravity', 'images': images,
             'prompt': rubric + f'\nAssigned case_id: {case_id}. Actual frame_count:97. Actual sheets:5.\n',
             'frame_count': 97, 'sheets': payload['sheets']})
        cases.append({'id': case_id, 'kind': 'gravity', 'split': 'i2v', 'input': str(folder / 'input.json')})
    save(ROOT / 'gravity_judge_cases.json', cases)
    save(ROOT / 'gravity_private_mapping.json', mapping)
    save(ROOT / 'judge_input_receipt.json', {'cases': len(cases), 'images_and_task_only': True,
         'rubric_sha256': sha(gravity / 'JUDGE_RUBRIC.md'), 'old_predictions_or_opinions_in_prompt': False})
    print(json.dumps({'gravity_calibration': 12, 'i2v': 4}))


def prepare_pc():
    plan = json.loads((ROOT / 'human_probe_plan.json').read_text())['rows']
    cases = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for row in plan:
        receipt_path = ROOT / 'human_videos' / (row['opaque_id'] + '.json')
        retry_path = ROOT / 'human_videos_direct' / (row['opaque_id'] + '.json')
        receipt = json.loads((retry_path if retry_path.exists() else receipt_path).read_text())
        folder = ROOT / 'judge_inputs_pc_v2' / row['opaque_id']
        folder.mkdir(parents=True, exist_ok=False)
        item = {'id': row['opaque_id'], 'kind': 'pc', 'split': row['split'], 'input': str(folder / 'input.json')}
        if receipt['status'] != 'ok':
            save(folder / 'input.json', {'id': row['opaque_id'], 'unavailable': True, 'reason': receipt['status']})
            cases.append(item)
            continue
        assert sha(receipt['file']) == receipt['sha256']
        cap = cv2.VideoCapture(receipt['file'])
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        indices = np.rint(np.linspace(0, len(frames) - 1, 32)).astype(int).tolist()
        images = []
        for offset in [0, 16]:
            image = Image.new('RGB', (4 * 256, 4 * 216), 'white')
            draw = ImageDraw.Draw(image)
            for i, index in enumerate(indices[offset:offset + 16]):
                small = Image.fromarray(frames[index])
                small.thumbnail((256, 192))
                x, y = (i % 4) * 256, (i // 4) * 216
                image.paste(small, (x, y + 24))
                draw.text((x + 3, y + 2), f'frame {index}', fill='black', font=font)
            path = folder / f'sheet_{offset}.png'
            image.save(path)
            images.append({'path': path.name, 'sha256': sha(path)})
        save(folder / 'input.json', {'id': row['opaque_id'], 'kind': 'pc', 'images': images,
             'prompt': PC_TEMPLATE.replace('{id}', row['opaque_id']), 'frame_indices': indices,
             'source_video_sha256': receipt['sha256'], 'source_frame_count': len(frames)})
        cases.append(item)
    save(ROOT / 'pc_judge_cases.json', cases)
    print(json.dumps({'pc_cases': len(cases), 'sampled_frames_per_video': 32}))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'pc':
        prepare_pc()
    else:
        main()
