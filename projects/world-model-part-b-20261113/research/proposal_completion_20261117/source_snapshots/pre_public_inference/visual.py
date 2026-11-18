"""Validate external visual observations without treating them as reference labels."""
import hashlib
import json
from pathlib import Path

from .contracts import sha


def opaque_id(case_id):
    return 'v_' + hashlib.sha256(case_id.encode()).hexdigest()[:16]


def index_opinions(opinions, allowed_ids):
    result = {}
    for opinion in opinions:
        identifier = opinion['opaque_case_id']
        if identifier in result:
            raise ValueError('duplicate visual case ID')
        if identifier not in allowed_ids:
            raise ValueError('unknown visual case ID')
        result[identifier] = opinion
    return result


def validate_opinion(opinion, payload_path):
    path = Path(payload_path)
    if sha(path) != opinion['payload_sha256']:
        raise ValueError('visual payload checksum mismatch')
    payload = json.loads(path.read_text())
    if opinion['opaque_case_id'] != payload['opaque_case_id']:
        raise ValueError('visual payload identity mismatch')
    expected = [sheet['sha256'] for sheet in payload['sheets']]
    if not expected or opinion['viewed_sheet_sha256'] != expected:
        raise ValueError('incomplete or wrong viewed sheet list')
    indices = []
    for sheet in payload['sheets']:
        image = (path.parent / sheet['path']).resolve()
        if not image.is_relative_to(path.parent.resolve()) or sha(image) != sheet['sha256']:
            raise ValueError('visual image checksum/path mismatch')
        indices.extend(sheet['frame_indices'])
    if indices != list(range(payload['frames'])):
        raise ValueError('incomplete chronological sheet frame coverage')
    verdict = opinion['verdict']
    if verdict not in ['PASS', 'FAIL', 'ABSTAIN'] or not isinstance(opinion['reason'], str) or not opinion['reason'].strip():
        raise ValueError('invalid visual verdict or reason')
    names = {'left_release_frame_range', 'right_release_frame_range',
             'left_contact_frame_range', 'right_contact_frame_range'}
    if set(opinion['events']) != names:
        raise ValueError('wrong event schema')
    for event in opinion['events'].values():
        if event is None:
            if verdict != 'ABSTAIN':
                raise ValueError('conclusive verdict lacks event evidence')
        elif not isinstance(event, list) or len(event) != 2 or not all(type(t) is int for t in event) or not 0 <= event[0] <= event[1] < payload['frames']:
            raise ValueError('invalid event evidence range')
    assumptions = opinion['assumptions']
    if set(assumptions) != {'synchronized_release', 'static_camera', 'both_falls_observed'}:
        raise ValueError('wrong assumption schema')
    if any(v not in ['supported', 'violated', 'unclear'] for v in assumptions.values()):
        raise ValueError('invalid assumption evidence')
    if verdict != 'ABSTAIN' and set(assumptions.values()) != {'supported'}:
        raise ValueError('conclusive verdict lacks admissibility evidence')
    origin = opinion['provenance']
    if origin.get('kind') not in ['human', 'isolated_ai', 'context_aware_ai'] or not origin.get('judge_id'):
        raise ValueError('missing visual provenance')
    # Content/hash validation alone cannot verify actual blinding or human origin.
    return {'status': 'diagnostic', 'verdict': verdict, 'reason': opinion['reason'],
            'events': opinion['events'], 'assumptions': assumptions, 'provenance': origin,
            'payload_sha256': opinion['payload_sha256'], 'viewed_sheet_sha256': expected,
            'independent_validation': False, 'ground_truth_label': False,
            'independence_status': 'not_independently_audited'}
