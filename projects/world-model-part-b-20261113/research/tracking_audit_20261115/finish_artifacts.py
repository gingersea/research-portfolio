"""Build descriptive figures and verify the immutable source experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from collect_figures import FIGURES
from diagnose import sha

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'principia_i2v_20261115'


def main():
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=False)
    rows = json.loads((ROOT / 'reanalysis_v1/summary.json').read_text())['cases']
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 9})
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4))
    original = np.array([row['original_eligible_frames'] for row in rows])
    new = np.array([row['candidate_eligible_frames'] for row in rows])
    ax.bar(x-.18, original/97, width=.36, label='Original frozen hue instrument', color='#888888')
    ax.bar(x+.18, new/97, width=.36, label='Exploratory foreground/association v1', color='#377eb8')
    for index in range(4):
        ax.text(index-.18, original[index]/97+.025, f'{original[index]}/97', ha='center', fontsize=8)
        ax.text(index+.18, new[index]/97+.025, f'{new[index]}/97', ha='center', fontsize=8)
    ax.axhline(.95, color='#b2182b', linestyle='--', label='Original minimum fraction (other gates also required)')
    ax.set(xticks=x, xticklabels=[row['clip'] for row in rows], ylim=(0, 1.16), ylabel='Both-object eligible component fraction',
           title='All 4 seen development clips; availability is not tracking accuracy')
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'coverage_comparison.{extension}', dpi=180)
    plt.close(fig)
    before = json.loads((ROOT / 'original_before.sha256.json').read_text())
    assert all(sha(OLD / name) == digest for name, digest in before.items())
    geometry = {
        'intrinsics': [[500, 0, 384], [0, 500, 256], [0, 0, 1]], 'pixel': [434, 281],
        'depths_m': [2, 4], 'points_camera_m': [[.2, .1, 2], [.4, .2, 4]],
        'world_point_m': [1.2, .1, 2], 'camera_centers_m': [[1, 0, 0], [1.1, 0, 0]],
        'horizontal_pixels_after_camera_move': [434, 409],
        'scope': 'Self-authored geometric teaching calculations, not observed video reconstruction.'}
    for point in geometry['points_camera_m']:
        projected = np.array(geometry['intrinsics']) @ point
        assert np.allclose(projected[:2]/projected[2], geometry['pixel'])
    (output / 'geometry_examples.json').write_text(json.dumps(geometry, indent=2) + '\n')
    figures = [{'paper': name, 'figure': figure, 'image_available': (ROOT / 'papers' / name / f'{figure}.png').exists(),
                'sha256': sha(ROOT / 'papers' / name / f'{figure}.png') if (ROOT / 'papers' / name / f'{figure}.png').exists() else None}
               for name, figure in FIGURES]
    for path in (ROOT / 'papers').glob('*/paper.html'):
        assert path.stat().st_size > 1000
    sources = json.loads((ROOT / 'paper_sources.json').read_text()) + json.loads((ROOT / 'paper_sources_updates.json').read_text())
    for source in sources:
        assert sha(ROOT / 'papers' / source['name'] / 'paper.html') == source['html_sha256']
        assert sha(ROOT / 'papers' / source['name'] / 'paper.txt') == source['text_sha256']
    controls = json.loads((ROOT / 'controls_v1/results.json').read_text())
    challenge = json.loads((ROOT / 'identity_challenge_v1/result.json').read_text())
    assert sha(ROOT / 'candidate.py') == controls['candidate_sha256'] == challenge['candidate_sha256']
    record = {'original_files_unchanged': len(before), 'original_before_after_sha256_passed': True,
              'paper_versions_verified': len(sources), 'figure_reading_availability': figures,
              'figure_download_warning': 'Initial collector exited143 after four figures were saved; single missing SpatialTracker figure retry reached explicit25s timeout(exit124). No claim of viewing the missing image.',
              'control_cases_passed': sum(row['passed'] for row in controls['cases']), 'control_cases_total': controls['controls'],
              'identity_challenge_passed': challenge['passed_identity_safety'],
              'wrong_identity_observations_in_challenge': challenge['wrong_B_identity_claims'],
              'candidate_is_production_ready': False, 'no_generated_video_regeneration': True,
              'python': platform.python_version(),
              'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'opencv-python', 'matplotlib', 'Pillow', 'requests', 'beautifulsoup4', 'pytest']}}
    (output / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
