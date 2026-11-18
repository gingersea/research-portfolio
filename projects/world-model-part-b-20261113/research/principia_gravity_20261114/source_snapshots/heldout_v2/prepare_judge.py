"""Export prespecified first two held-out scenes, all conditions, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    truth = json.loads((args.data / 'private/heldout_truth.json').read_text())
    selected = {row['case_id'] for row in truth if row['scene_seed'] in [81000, 81001]}
    assert len(selected) == 12
    rows = json.loads((args.data / 'heldout_public_index.json').read_text())
    index = []
    for case in sorted(rows, key=lambda row: row['case_id']):
        identifier = case['case_id']
        if identifier not in selected:
            continue
        folder = STUDY / 'judge_inputs' / identifier
        folder.mkdir(parents=True, exist_ok=False)
        setup = json.loads((args.data / case['setup_path']).read_text())
        clip = args.data / case['videos']['ffv1']['path']
        assert sha(clip) == case['videos']['ffv1']['sha256']
        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        assert len(frames) == setup['frames']
        sheets = []
        for offset in range(0, len(frames), 24):
            sheet = Image.new('RGB', (6 * 256, 4 * 212), 'white')
            draw = ImageDraw.Draw(sheet)
            included = []
            for position, frame in enumerate(frames[offset:offset + 24]):
                x, y = (position % 6) * 256, (position // 6) * 212
                sheet.paste(Image.fromarray(frame).resize((256, 192)), (x, y))
                frame_index = offset + position
                included.append(frame_index)
                draw.text((x + 4, y + 194), f'frame {frame_index:03d} | {frame_index/setup["fps"]:.3f} sec', fill='black')
            path = folder / f'frames_{offset // 24:02d}.png'
            sheet.save(path)
            sheets.append({'path': path.name, 'frame_indices': included, 'sha256': sha(path)})
        shutil.copy2(STUDY / 'JUDGE_RUBRIC.md', folder / 'rubric.md')
        payload = {'case_id': identifier, 'fps': setup['fps'], 'frame_count': len(frames), 'setup': setup['setup'], 'scope': 'Initial fall of two balls; no bouncing. Apply the assigned rubric; all native frames are present in chronological sheets.', 'sheets': sheets, 'output_path': str(STUDY / 'judgments' / f'{identifier}.json')}
        (folder / 'payload.json').write_text(json.dumps(payload, indent=2) + '\n')
        assert [frame for sheet in sheets for frame in sheet['frame_indices']] == list(range(len(frames)))
        index.append({'case_id': identifier, 'public_input_dir': str(folder), 'payload_sha256': sha(folder / 'payload.json'), 'rubric_sha256': sha(folder / 'rubric.md'), 'sheet_count': len(sheets), 'native_frame_count': len(frames), 'source_ffv1_sha256': sha(clip)})
    (STUDY / 'judgments').mkdir(exist_ok=True)
    target = STUDY / 'judge_input_manifest.json'
    if target.exists():
        raise FileExistsError('Refuse to replace judge assignment')
    target.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'anonymous_cases': len(index), 'sheets': sum(row['sheet_count'] for row in index), 'native_frames': sum(row['native_frame_count'] for row in index)}))


if __name__ == '__main__':
    main()
