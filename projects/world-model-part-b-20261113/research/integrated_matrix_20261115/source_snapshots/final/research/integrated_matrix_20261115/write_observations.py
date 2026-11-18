"""Record the current agent's actually viewed, nonblind RGB observations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.relations.contracts import save, sha
from physcheck.relations.visual import validate_opinion

REASONS = {
    'C1': 'The red shape fades during frames 3-16, and a red shape appears near the blue ball in frames 17-24 before disappearing again. The blue shape remains above the floor and later changes appearance. Two identifiable complete release-to-contact falls are not observed.',
    'C2': 'Both balls remain approximately at their initial heights through frame 61. The blue ball fades around frames 62-65 and later small green features appear near the red ball; the red ball remains suspended. Neither two releases nor two first floor contacts can be localized.',
    'C3': 'The left red object changes shape in the first few frames, becomes pale, and later becomes cyan/blue around frames 54-56. The right blue object remains above the floor. No pair of identifiable complete initial falls ending at the floor is visible.',
    'C4': 'Both colored shapes move downward during the first few frames, deform near the floor around frames 3-8, and disappear by the following frames. A stable identifiable first contact for each ball cannot be established; frames 9-96 do not recover the two balls.'}


def main():
    opinions, viewed = [], []
    prior = PROJECT / 'research/integrated_framework_20261115/fresh_i2v4_final'
    for clip, reason in REASONS.items():
        folder = prior / ('i2v_' + clip) / 'blind_visual_input'
        path = folder / 'visual_judge_payload.json'
        payload = json.loads(path.read_text())
        assert [s['path'] for s in payload['sheets']] == ['rgb_000.png', 'rgb_024.png', 'rgb_048.png', 'rgb_072.png', 'rgb_096.png']
        events = {key: None for key in ['left_release_frame_range', 'right_release_frame_range',
                                      'left_contact_frame_range', 'right_contact_frame_range']}
        if clip == 'C4':
            events['left_release_frame_range'] = [0, 2]
            events['right_release_frame_range'] = [0, 2]
        opinion = {'opaque_case_id': payload['opaque_case_id'], 'payload_sha256': sha(path),
             'viewed_sheet_sha256': [s['sha256'] for s in payload['sheets']],
             'verdict': 'ABSTAIN', 'reason': reason, 'events': events,
             'assumptions': {'synchronized_release': 'unclear', 'static_camera': 'supported', 'both_falls_observed': 'violated'},
             'provenance': {'kind': 'context_aware_ai', 'judge_id': 'current_assistant_session_exact_model_version_not_recorded',
                  'prior_context_seen': True, 'human_label': False,
                  'observation_method': 'all listed RGB sheets actually viewed using tools.view_image',
                  'blinding_limit': 'The observer had already read numerical and qualitative study results; not an isolated judge.'}}
        validate_opinion(opinion, path)
        opinions.append(opinion)
        for sheet in payload['sheets']:
            viewed.append({'clip': clip, 'path': str(folder / sheet['path']), 'sha256': sheet['sha256'],
                           'frame_indices': sheet['frame_indices'], 'actually_viewed_in_current_turn': True})
    save(ROOT / 'visual_opinions.json', opinions)
    save(ROOT / 'visual_observation_receipt.json', {'recorded_utc': datetime.now(timezone.utc).isoformat(),
         'actually_viewed_sheets': viewed, 'viewed_total': 20, 'source_sha256': sha(Path(__file__)),
         'opinions_sha256': sha(ROOT / 'visual_opinions.json'), 'independent_blind_judging': False,
         'human_ground_truth': False, 'selection': 'all four existing clips, all chronological sheets'})
    print(json.dumps({'opinions': 4, 'actually_viewed_sheets': 20, 'independent': False}))


if __name__ == '__main__':
    main()
