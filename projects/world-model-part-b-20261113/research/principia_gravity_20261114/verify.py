"""Read-only artifact verification and fresh image-only replay on two full scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from assess_measurement import assess
from judge_gate import validate
from measure import measure_video

STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    if args.record.exists():
        raise FileExistsError('Refuse to replace verification evidence')
    snapshot = json.loads((STUDY / 'source_snapshots/heldout_v2/manifest.json').read_text())
    for name, expected in snapshot['source_sha256'].items():
        assert sha(STUDY / name) == expected, name
    truth = json.loads((STUDY / 'data/private/heldout_truth.json').read_text())
    measurements = json.loads((STUDY / 'heldout_measurements_v2.json').read_text())
    saved_gate = json.loads((STUDY / 'heldout_gate_v2.json').read_text())
    recomputed = assess(measurements, truth)
    for key in recomputed:
        assert recomputed[key] == saved_gate[key], key
    observed = {(row['case_id'], row['codec']): row for row in measurements}
    replay_ids = {row['case_id'] for row in truth if row['scene_seed'] in [81000, 81001]}
    media_count = 0
    replay_count = 0
    for split in ['development', 'heldout']:
        index = json.loads((STUDY / 'data' / f'{split}_public_index.json').read_text())
        for case in index:
            setup = json.loads((STUDY / 'data' / case['setup_path']).read_text())
            for codec, entry in case['videos'].items():
                clip = STUDY / 'data' / entry['path']
                assert sha(clip) == entry['sha256']
                media_count += 1
                if split == 'heldout' and case['case_id'] in replay_ids:
                    actual = measure_video(clip, setup)
                    assert {**{'case_id': case['case_id'], 'codec': codec}, **actual} == observed[(case['case_id'], codec)]
                    replay_count += 1
    manifest = json.loads((STUDY / 'judge_input_manifest.json').read_text())
    receipts = json.loads((STUDY / 'judge_session_receipts.json').read_text())['receipts']
    for case in manifest:
        directory = Path(case['public_input_dir'])
        payload = json.loads((directory / 'payload.json').read_text())
        assert sha(directory / 'payload.json') == case['payload_sha256']
        assert sha(directory / 'rubric.md') == case['rubric_sha256'] == sha(STUDY / 'JUDGE_RUBRIC.md')
        frames = []
        for sheet in payload['sheets']:
            assert sha(directory / sheet['path']) == sheet['sha256']
            frames.extend(sheet['frame_indices'])
        assert frames == list(range(payload['frame_count']))
        judgment = STUDY / 'judgments' / f'{case["case_id"]}.json'
        validate(json.loads(judgment.read_text()), payload)
        assert sha(judgment) == receipts[case['case_id']]['judgment_sha256']
        assert receipts[case['case_id']]['all_assigned_images_viewed']
        assert not receipts[case['case_id']]['private_or_outside_input_access']
    official = json.loads((STUDY / 'official_crosscheck.json').read_text())
    assert np.isclose(official['ordering_counterexample']['native_unordered_score'], 1)
    assert np.isclose(official['ordering_counterexample']['directed_score'], .4)
    result = {'passed': True, 'source_snapshot_sha256_matched': len(snapshot['source_sha256']), 'encoded_media_hashes_verified': media_count, 'fresh_image_only_case_codec_replays': replay_count, 'measurement_gate_recomputed': recomputed['passed'], 'judge_inputs_and_first_results_verified': len(manifest), 'all_native_judge_frame_indices_present': sum(row['native_frame_count'] for row in manifest), 'judge_images_verified': sum(row['sheet_count'] for row in manifest), 'data_truth_sha256': sha(STUDY / 'data/private/heldout_truth.json')}
    args.record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
