"""Verify all six saved native-frame outputs; create unblinded inspection sheets."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

STUDY = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    records = []
    noise = set()
    configs = []
    sheets = STUDY / 'inspection'
    sheets.mkdir(exist_ok=True)
    arrays = {}
    for arm, scales in [('base', [2.2, 3.2]), ('vpt', [2.2])]:
        folder = STUDY / 'generation' / arm
        protocol = json.loads((folder / 'protocol.json').read_text())
        configs.append(protocol)
        assert (protocol['frames'], protocol['height'], protocol['width'], protocol['steps']) == (81, 480, 832, 50)
        assert protocol['scheduler']['shift'] == 5.0
        noise.add(protocol['rgb_noise_bf16_sha256'])
        for prompt in ['bounce', 'collision']:
            for scale in scales:
                stem = f'{prompt}_seed42_gs{scale}'
                video = folder / f'{stem}.mp4'
                native = folder / f'{stem}_frames.npz'
                record = json.loads((folder / f'{stem}.json').read_text())
                assert sha(video) == record['sha256']
                assert record['initial_rgb_noise_sha256'] == protocol['rgb_noise_bf16_sha256']
                assert record['prompt'] == protocol['prompts'][prompt]
                assert record['prepare_latents_calls'] == (3 if arm == 'vpt' else 1)
                assert record['effective_text_cfg'] == (3.2 if arm == 'vpt' else scale)
                pixels = np.load(native, allow_pickle=False)['frames']
                assert pixels.shape == (81, 480, 832, 3) and pixels.dtype == np.uint8
                cap = cv2.VideoCapture(str(video))
                assert cap.isOpened()
                fps = cap.get(cv2.CAP_PROP_FPS)
                assert abs(fps - 16) < 1e-6
                count = 0
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    assert frame.shape == (480, 832, 3)
                    count += 1
                cap.release()
                assert count == 81
                # These are content diagnostics, not measurements of physical law.
                black = int(np.all(pixels == 0, axis=(1, 2, 3)).sum())
                white = int(np.all(pixels == 255, axis=(1, 2, 3)).sum())
                delta = float(np.abs(pixels[1:].astype(np.float32) - pixels[:-1].astype(np.float32)).mean())
                assert black == 0 and white == 0 and delta > 0
                sheet = Image.new('RGB', (9 * 208, 9 * 140), 'white')
                draw = ImageDraw.Draw(sheet)
                for index, frame in enumerate(pixels):
                    x, y = (index % 9) * 208, (index // 9) * 140
                    sheet.paste(Image.fromarray(frame).resize((208, 120)), (x, y))
                    draw.text((x + 3, y + 122), f'{arm} gs{scale} f{index} t{index/16:.3f}', fill='black')
                sheet_path = sheets / f'{arm}_{stem}_all81.jpg'
                sheet.save(sheet_path, quality=95)
                arrays[(arm, scale, prompt)] = pixels
                records.append({'arm': arm, 'prompt_id': prompt, 'scale': scale, 'decoded_frames': count, 'fps': fps, 'shape': list(pixels.shape), 'mp4_sha256': record['sha256'], 'native_frames_sha256': sha(native), 'all81_sheet': str(sheet_path.relative_to(STUDY)), 'all_black_frames': black, 'all_white_frames': white, 'adjacent_pixel_mae_diagnostic_only': delta, 'elapsed_seconds': record['elapsed_seconds']})
    assert len(noise) == 1 and len(records) == 6
    for key in ['prompts', 'seed', 'height', 'width', 'frames', 'steps', 'fps', 'scheduler', 'negative_prompt_sha256', 'base_transformer_config']:
        left, right = configs[0][key], configs[1][key]
        if key == 'scheduler':
            # Diffusers serializes a set of defaulted parameter names into a list;
            # Python hash randomization changes only that list's order per process.
            left = {k: sorted(v) if k == '_use_default_values' else v for k, v in left.items()}
            right = {k: sorted(v) if k == '_use_default_values' else v for k, v in right.items()}
        assert left == right, key
    assert len({r['mp4_sha256'] for r in records}) == 6
    for prompt in ['bounce', 'collision']:
        sheet = Image.new('RGB', (3 * 312, 7 * 200), 'white')
        draw = ImageDraw.Draw(sheet)
        for col, (arm, scale) in enumerate([('base', 2.2), ('base', 3.2), ('vpt', 2.2)]):
            draw.text((col * 312 + 5, 5), f'{arm} parameter={scale} effective CFG={3.2 if arm == "vpt" else scale}', fill='black')
            for row, index in enumerate([0, 8, 16, 32, 48, 64, 80]):
                x, y = col * 312, row * 195 + 25
                sheet.paste(Image.fromarray(arrays[(arm, scale, prompt)][index]).resize((312, 180)), (x, y))
                draw.text((x + 4, y + 180), f'f{index} t{index/16:.2f}s', fill='black')
        sheet.save(sheets / f'{prompt}_comparison.jpg', quality=95)
    result = {'production_videos': len(records), 'shared_rgb_noise_sha256': next(iter(noise)), 'checks': 'All expected records present, hashes match, native and decoded shape/count/FPS correct, shared solver/settings/prompts/noise, distinct outputs, no fully black/white frames.', 'physical_improvement_claim': False, 'new_human_labels': 0, 'calibrated_automatic_judge': False, 'records': records, 'environment': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'diffusers', 'peft', 'huggingface-hub', 'tokenizers', 'accelerate', 'numpy', 'opencv-python']}, 'runner_sha256': sha(STUDY / 'run_vpt.py')}
    (STUDY / 'generation_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['records', 'environment']}, indent=2))


if __name__ == '__main__':
    main()
