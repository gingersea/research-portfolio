"""Strict joins and calibration-gated Part B aggregation for production runs."""
import hashlib
import json
from typing import NotRequired, TypedDict

from physcheck.judge.calibration import check_calibration, ProtocolInvalid
from physcheck.judge.protocol import parse_judge_output
from physcheck.report.aggregate import aggregate_part_b


class ManifestEntry(TypedDict):
    model: str
    prompt_id: str
    seed: int
    kind: str
    is_calibration: bool
    sampled_frame_indices: list[int]
    expected_violation_kind: NotRequired[str | None]
    calibration_observable: NotRequired[bool]


class JudgeInput(TypedDict):
    video_id: str
    raw_output: str
    judge_model: str
    judge_version: str
    judge_session: str


class Violation(TypedDict):
    frame_idx: int
    category: str
    description: str
    confidence: float


class JudgeOutput(TypedDict):
    video_id: str
    violations: list[Violation]
    notes: str
    judge_model: str
    judge_version: str
    judge_session: str


class CalibrationGate(TypedDict):
    valid: bool
    reason: NotRequired[str]
    failed_video_ids: NotRequired[list[str]]
    unverifiable_controls: NotRequired[list[str]]


class ValidationCheckpoint(TypedDict):
    manifest_sha256: str
    calibration: CalibrationGate
    validated: list[JudgeOutput]
    calibration_only: bool
    judge_model: str
    judge_version: str


def _manifest_digest(manifest: dict[str, ManifestEntry]) -> str:
    canonical=json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_records(records: list[JudgeInput], manifest: dict[str, ManifestEntry],
                     calibration_only: bool=False) -> ValidationCheckpoint:
    expected={vid for vid,info in manifest.items() if not calibration_only or info['is_calibration']}
    ids=[r['video_id'] for r in records]
    if len(ids)!=len(set(ids)):
        raise ValueError('duplicate video judgments')
    if set(ids)!=expected:
        raise ValueError(f'coverage mismatch: missing {sorted(expected-set(ids))}, extra {sorted(set(ids)-expected)}')
    identities={(r.get('judge_model'),r.get('judge_version')) for r in records}
    if len(identities)!=1 or any(not v or v=='unknown' for pair in identities for v in pair):
        raise ValueError('one known judge model/version is required')
    sessions=[r.get('judge_session') for r in records]
    if not all(sessions) or len(set(sessions))!=len(sessions):
        raise ValueError('each clip requires an independent judge session')
    parsed={}
    for record in records:
        vid=record['video_id'];info=manifest[vid]
        out=parse_judge_output(record['raw_output'],vid,record['judge_model'],record['judge_version'])
        text=record['raw_output'].strip()
        if text.startswith('```'):
            text=text.split('\n',1)[1].rsplit('```',1)[0]
        data=json.loads(text)
        if data.get('video_id')!=vid:
            raise ValueError(f'embedded video_id mismatch: {vid}')
        if not isinstance(data.get('notes'),str) or any(
            not isinstance(v.get('description'),str) for v in data['violations']
        ):
            raise ValueError(f'nontext field violates judge schema: {vid}')
        if any(isinstance(v[field],bool) for v in data['violations'] for field in ['frame_idx','confidence']):
            raise ValueError(f'boolean is not a numeric judgment: {vid}')
        for violation in out['violations']:
            if violation['frame_idx'] not in info['sampled_frame_indices']:
                raise ValueError(f'violation references unsampled frame: {vid}')
            if not violation['description'].strip():
                raise ValueError(f'empty violation description: {vid}')
        out['judge_session']=record['judge_session']
        parsed[vid]=out
    controls={vid:info for vid,info in manifest.items() if info['is_calibration']}
    if not controls:
        raise ValueError('calibration controls are required')
    for vid,info in controls.items():
        if 'calibration_observable' in info and type(info['calibration_observable']) is not bool:
            raise ValueError(f'control eligibility must be boolean: {vid}')
    try:
        gate=check_calibration(parsed,controls)
    except ProtocolInvalid as error:
        gate={'valid':False,'reason':str(error)}
    unverified=[vid for vid,info in controls.items() if info.get('calibration_observable') is not True]
    if unverified:
        gate={**gate,'valid':False,'unverifiable_controls':unverified}
    return {'manifest_sha256':_manifest_digest(manifest),
            'calibration':gate,'validated':list(parsed.values()),'calibration_only':calibration_only,
            'judge_model':next(iter(identities))[0],'judge_version':next(iter(identities))[1]}


def build_leaderboard(validation: ValidationCheckpoint,manifest: dict[str, ManifestEntry]) -> dict:
    if validation.get('manifest_sha256')!=_manifest_digest(manifest):
        raise ValueError('checkpoint manifest binding missing or mismatched')
    gate=validation.get('calibration')
    if not isinstance(gate,dict) or type(gate.get('valid')) is not bool:
        raise ValueError('checkpoint gate must contain a boolean valid flag')
    if validation.get('calibration_only') is not False:
        raise ValueError('production checkpoint must have calibration_only=false')
    # Recheck persisted rows at the report boundary, rather than trusting a
    # saved success boolean or silently filtering incomplete coverage.
    records=[{'video_id':row['video_id'],'raw_output':json.dumps(row,allow_nan=False),
              'judge_model':row['judge_model'],'judge_version':row['judge_version'],
              'judge_session':row['judge_session']} for row in validation['validated']]
    checked=validate_records(records,manifest)
    if checked['calibration']['valid'] is not gate['valid']:
        raise ValueError('checkpoint gate is inconsistent with retained judgments')
    if any(checked[key]!=validation.get(key) for key in ['judge_model','judge_version']):
        raise ValueError('checkpoint judge identity is inconsistent with retained judgments')
    validation=checked
    production={vid:info for vid,info in manifest.items() if not info['is_calibration']}
    outputs=[r for r in validation['validated'] if r['video_id'] in production]
    diagnostics=aggregate_part_b(outputs,production)
    if validation['calibration']['valid']:
        return {'valid':True,'calibration':validation['calibration'],**diagnostics}
    return {'valid':False,'calibration':validation['calibration'],
            'violation_rate_by_model':None,'mean_violations_per_video':None,
            'diagnostic_only':diagnostics,
            'reason':'Calibration failed or controls were not verifiable; no valid ranking.'}
