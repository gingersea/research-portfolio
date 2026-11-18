"""Postproduction descriptive artifacts; does not change the frozen gate results."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluate import assess_frames, component

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = STUDY / 'summary'
    output.mkdir(exist_ok=False)
    result = json.loads((STUDY / 'pilot_gate.json').read_text())
    frozen = json.loads((STUDY / 'source_snapshots/production_frozen/manifest.json').read_text())
    assert all(sha(STUDY / name) == digest for name, digest in frozen['sha256'].items())
    setups = {row['initial_id']: row for row in json.loads((STUDY / 'input_manifest.json').read_text())}
    metrics, frame_rows, integrity, vae = [], [], [], []
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(4, 7, figsize=(14, 6.5))
    initial_fig, initial_axes = plt.subplots(2, 3, figsize=(10, 5))
    for row_index, row in enumerate(result['cases']):
        identifier = f"{row['initial_id']}_seed{row['seed']}"
        folder = STUDY / 'generation/primary' / row['initial_id']
        receipt = json.loads((folder / f'{identifier}.json').read_text())
        protocol = json.loads((folder / 'executed_protocol.json').read_text())
        pixels = np.load(STUDY / receipt['array'], allow_pickle=False)['frames']
        first_path = folder / f'{identifier}_actual_first.png'
        # The runner's saved first frame is checked against the actual frame array.
        if not first_path.exists():
            first_path = folder / f'{identifier}_first.png'
        assert sha(first_path) == receipt['actual_first_frame_sha256']
        assert np.array_equal(np.array(Image.open(first_path).convert('RGB')), pixels[0])
        assert protocol['runner_sha256'] == frozen['sha256']['run_i2v.py']
        setup = setups[row['initial_id']]
        assert protocol['input_sha256'] == sha(STUDY / setup['image'])
        assert receipt['conditioning']['conditioned_tokens'] == 384
        raw = cv2.VideoCapture(str(STUDY / 'inspection' / f'{identifier}_native.mkv'))
        count = 0
        while True:
            ok, frame = raw.read()
            if not ok:
                break
            assert np.array_equal(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pixels[count])
            count += 1
        raw.release()
        assert count == 97
        integrity.append({'clip': identifier, 'lossless_frames_equal': count, 'first_frame_matches_array': True,
                          'frozen_runner_matches': True, 'input_hash_matches': True})
        masks_file = np.load(STUDY / 'inputs' / row['initial_id'] / 'masks.npz', allow_pickle=False)
        masks = [masks_file['left'], masks_file['right']]
        for frame_index, frame in enumerate(pixels):
            statuses = []
            for index, color in enumerate(setup['colors_rgb_left_right']):
                found = component(frame, color)
                status = 'missing_or_ambiguous' if found is None else (
                    'truncated' if not found['complete'] else (
                        'area_out_of_range' if not .75 <= found['area'] / masks[index].sum() <= 1.25 else 'usable'))
                statuses.append(status)
            frame_rows.append({'clip': identifier, 'frame': frame_index, 'time_seconds': frame_index / 24,
                               'left_component_status': statuses[0], 'right_component_status': statuses[1],
                               'both_usable': row['both_usable_by_frame'][frame_index]})
        metrics.append({'clip': identifier, **{key: row[key] for key in [
            'initial_objects_passed', 'initial_setup_passed', 'usable_both_frames', 'expected_frames',
            'usable_both_fraction', 'longest_missing_run', 'coverage_passed',
            'individual_fiducial_camera_passed', 'fiducial_unavailable_frames',
            'floor_observability_passed', 'frozen_measurement_reason', 'physics_relation_available', 'physics_verdict']}})
        for column, frame_index in enumerate([0, 1, 12, 24, 48, 72, 96]):
            axes[row_index, column].imshow(pixels[frame_index])
            axes[row_index, column].set_title(f'{identifier}, f{frame_index} ({frame_index/24:.2f}s)', fontsize=8)
            axes[row_index, column].axis('off')
        initial_row = 0 if row['initial_id'] == 'A' else 1
        if row['seed'] == 42:
            initial_axes[initial_row, 0].imshow(Image.open(STUDY / setup['image']))
            initial_axes[initial_row, 0].set_title(f"Input {row['initial_id']}")
        column = 1 if row['seed'] == 42 else 2
        initial_axes[initial_row, column].imshow(pixels[0])
        initial_axes[initial_row, column].set_title(f'{identifier}: actual generated f0')
    for axis in initial_axes.flat:
        axis.axis('off')
    fig.suptitle('All four fixed attempts; identical frame selection, no output selection')
    initial_fig.suptitle('Authored input versus native generated first frame (no pixel replacement)')
    for figure, name in [(fig, 'fixed_frame_overview'), (initial_fig, 'first_frame_preservation')]:
        figure.tight_layout()
        for extension in ['png', 'pdf', 'svg']:
            figure.savefig(output / f'{name}.{extension}', dpi=180)
        plt.close(figure)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), gridspec_kw={'height_ratios': [2, 1.5]})
    labels = [row['clip'] for row in metrics]
    coverage = [row['usable_both_fraction'] for row in metrics]
    axes[0].bar(labels, coverage, color='#377eb8', width=.6)
    axes[0].axhline(.95, color='#b2182b', linestyle='--', label='Prespecified minimum 95%')
    for index, value in enumerate(coverage):
        axes[0].text(index, value+.02, f"{metrics[index]['usable_both_frames']}/97 ({value:.1%})", ha='center')
    axes[0].set(ylim=(0, 1.12), ylabel='Both-object component availability')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[1].imshow([row['both_usable_by_frame'] for row in result['cases']], aspect='auto', interpolation='nearest',
                   cmap=matplotlib.colors.ListedColormap(['#e5e5e5', '#377eb8']), vmin=0, vmax=1,
                   extent=[-.5, 96.5, 3.5, -.5])
    axes[1].set(yticks=range(4), yticklabels=labels, xlabel='Expected frame index (24 FPS); blue=usable, gray=unavailable')
    fig.suptitle('2 setups x 2 seed repeats; 97 frames per attempt, no independent-frame inference')
    fig.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'coverage.{extension}', dpi=180)
    plt.close(fig)
    for name, rows in [('clip_metrics.csv', metrics), ('frame_component_diagnostics.csv', frame_rows)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for identifier, setup in setups.items():
        masks = np.load(STUDY / 'inputs' / identifier / 'masks.npz', allow_pickle=False)
        pixels = np.array(Image.open(STUDY / 'vae_roundtrip/primary' / f'{identifier}_vae_mode.png').convert('RGB'))[None]
        measured = assess_frames(pixels, setup, [masks['left'], masks['right']], expected_frames=1)
        vae.append({'input': identifier, 'single_image_diagnostic_only': True,
                    'initial_objects_passed': measured['initial_objects_passed'],
                    'initial_setup_passed': measured['initial_setup_passed']})
    versions = {name: importlib.metadata.version(name) for name in [
        'torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'huggingface-hub',
        'numpy', 'scipy', 'Pillow', 'opencv-python', 'matplotlib']}
    (output / 'artifact_verification.json').write_text(json.dumps({
        'production_snapshot_unchanged_files': len(frozen['sha256']), 'clips': integrity,
        'vae_mode_diagnostics': vae, 'analysis_python': platform.python_version(),
        'analysis_package_versions': versions,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True),
        'scope': 'Postproduction artifact integrity and descriptive diagnostics; no new model runs or threshold changes.'
    }, indent=2) + '\n')
    print(json.dumps({'frozen_sources_unchanged': len(frozen['sha256']), 'lossless_frames_verified': sum(x['lossless_frames_equal'] for x in integrity),
                      'initial_passed': sum(x['initial_objects_passed'] and x['initial_setup_passed'] for x in metrics),
                      'coverage_passed': sum(x['coverage_passed'] for x in metrics),
                      'usable_frames_total_diagnostic': sum(x['usable_both_frames'] for x in metrics),
                      'vae': vae}, indent=2))


if __name__ == '__main__':
    main()
