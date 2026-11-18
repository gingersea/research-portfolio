"""Real Part B joins must preserve blinding, coverage and failed calibration."""
import json
import pytest

from physcheck.judge.production import validate_records, build_leaderboard


def manifest():
    return {
        'opaque_control':{'model':'m','prompt_id':'control','seed':0,'kind':'support',
                          'is_calibration':True,'expected_violation_kind':'support',
                          'calibration_observable':True,
                          'sampled_frame_indices':[0,7,14,21,28,35,42,48]},
        'opaque_test':{'model':'m','prompt_id':'test','seed':0,'kind':'fluid',
                       'is_calibration':False,'sampled_frame_indices':[0,7,14,21,28,35,42,48]},
    }


def record(vid,violation=True):
    return {'video_id':vid,'raw_output':json.dumps({'video_id':vid,'violations':[
        {'frame_idx':7,'category':'support','description':'The unsupported object is visibly detached.','confidence':.9}
    ] if violation else [],'notes':''}),'judge_model':'fixed-model','judge_version':'fixed-version','judge_session':vid+'-session'}


def test_opaque_control_ids_validate_per_video():
    out=validate_records([record('opaque_control'),record('opaque_test',False)],manifest())
    assert out['calibration']['valid'] is True
    assert len(out['validated'])==2


def test_missing_control_never_disappears_from_denominator():
    with pytest.raises(ValueError,match='coverage'):
        validate_records([record('opaque_test')],manifest())


def test_duplicate_judgments_are_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        validate_records([record('opaque_control'),record('opaque_control'),record('opaque_test')],manifest())


def test_mismatched_embedded_id_is_rejected():
    bad=record('opaque_control');bad['raw_output']=bad['raw_output'].replace('opaque_control','wrong')
    with pytest.raises(ValueError,match='video_id'):
        validate_records([bad,record('opaque_test')],manifest())


def test_unsampled_frame_is_rejected():
    bad=record('opaque_control');bad['raw_output']=bad['raw_output'].replace('"frame_idx": 7','"frame_idx": 8')
    with pytest.raises(ValueError,match='frame'):
        validate_records([bad,record('opaque_test')],manifest())


def test_mixed_judge_versions_are_rejected():
    bad=record('opaque_test');bad['judge_version']='changed'
    with pytest.raises(ValueError,match='version'):
        validate_records([record('opaque_control'),bad],manifest())


def test_reused_judge_session_is_rejected():
    bad=record('opaque_test');bad['judge_session']='opaque_control-session'
    with pytest.raises(ValueError,match='session'):
        validate_records([record('opaque_control'),bad],manifest())


def test_failed_control_blocks_valid_scores_but_preserves_diagnostics():
    out=validate_records([record('opaque_control',False),record('opaque_test')],manifest())
    assert out['calibration']['valid'] is False
    board=build_leaderboard(out,manifest())
    assert board['valid'] is False
    assert board['violation_rate_by_model'] is None
    assert board['diagnostic_only']['violation_rate_by_model']['m']==1
    assert len(board['diagnostic_only']['per_video'])==1


def test_successful_calibration_excluded_from_production_rate():
    out=validate_records([record('opaque_control'),record('opaque_test',False)],manifest())
    board=build_leaderboard(out,manifest())
    assert board['valid'] is True
    assert board['violation_rate_by_model']['m']==0


def test_calibration_only_does_not_need_production_judgments():
    out=validate_records([record('opaque_control')],manifest(),calibration_only=True)
    assert out['calibration']['valid'] is True


def test_unobservable_control_cannot_pass_from_prompt_text():
    info=manifest();info['opaque_control']['calibration_observable']=False
    out=validate_records([record('opaque_control'),record('opaque_test')],info)
    assert out['calibration']['valid'] is False


def test_missing_control_eligibility_cannot_pass():
    info=manifest();del info['opaque_control']['calibration_observable']
    out=validate_records([record('opaque_control'),record('opaque_test')],info)
    assert out['calibration']['valid'] is False
    assert out['calibration']['unverifiable_controls']==['opaque_control']


@pytest.mark.parametrize('eligibility',['false',0,1,None])
def test_malformed_control_eligibility_is_rejected(eligibility):
    info=manifest();info['opaque_control']['calibration_observable']=eligibility
    with pytest.raises(ValueError,match='eligibility'):
        validate_records([record('opaque_control'),record('opaque_test')],info)


@pytest.mark.parametrize('change',['eligibility','coverage','model'])
def test_report_rejects_a_checkpoint_from_a_changed_manifest(tmp_path,change):
    from types import SimpleNamespace
    from physcheck.report.aggregate import main
    info=manifest()
    status=validate_records([record('opaque_control'),record('opaque_test',False)],info)
    if change=='eligibility':
        info['opaque_control']['calibration_observable']=False
    elif change=='coverage':
        info['missing_production']=dict(info['opaque_test'])
    else:
        info['opaque_test']['model']='changed-model'
    raw=tmp_path/'raw';raw.mkdir()
    (raw/'part_b_manifest.json').write_text(json.dumps(info))
    (raw/'judge_validated.status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError,match='manifest'):
        main(SimpleNamespace(results_dir=tmp_path),part='b')
    assert not (tmp_path/'leaderboard/part_b_leaderboard.json').exists()


@pytest.mark.parametrize('value',['false',1,None])
def test_report_rejects_a_nonboolean_gate(value):
    info=manifest()
    status=validate_records([record('opaque_control'),record('opaque_test')],info)
    status['calibration']['valid']=value
    with pytest.raises(ValueError,match='gate'):
        build_leaderboard(status,info)


@pytest.mark.parametrize('change',['missing','extra','duplicate'])
def test_report_rechecks_checkpoint_record_coverage(change):
    info=manifest()
    status=validate_records([record('opaque_control'),record('opaque_test')],info)
    if change=='missing':
        status['validated'].pop()
    else:
        extra=dict(status['validated'][-1])
        if change=='extra':
            extra['video_id']='unexpected'
        status['validated'].append(extra)
    with pytest.raises(ValueError,match='coverage|duplicate'):
        build_leaderboard(status,info)


def test_report_rechecks_gate_against_retained_judgments():
    info=manifest()
    status=validate_records([record('opaque_control',False),record('opaque_test')],info)
    status['calibration']={'valid':True,'failed_video_ids':[]}
    with pytest.raises(ValueError,match='gate'):
        build_leaderboard(status,info)


def test_strict_production_report_requires_its_status_checkpoint(tmp_path):
    from types import SimpleNamespace
    from physcheck.report.aggregate import main
    info=manifest()
    status=validate_records([record('opaque_control'),record('opaque_test',False)],info)
    raw=tmp_path/'raw';raw.mkdir()
    (raw/'part_b_manifest.json').write_text(json.dumps(info))
    (raw/'judge_validated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in status['validated']))
    with pytest.raises(ValueError,match='status'):
        main(SimpleNamespace(results_dir=tmp_path),part='b')
    assert not (tmp_path/'leaderboard/part_b_leaderboard.json').exists()


@pytest.mark.parametrize('field,value',[('frame_idx',False),('confidence',True)])
def test_boolean_is_not_a_numeric_judgment(field,value):
    bad=record('opaque_control');data=json.loads(bad['raw_output'])
    data['violations'][0][field]=value;bad['raw_output']=json.dumps(data)
    with pytest.raises(ValueError,match='numeric'):
        validate_records([bad,record('opaque_test')],manifest())


@pytest.mark.parametrize('location',["notes","description"])
def test_nontext_schema_fields_are_rejected_instead_of_coerced(location):
    bad=record('opaque_control');data=json.loads(bad['raw_output'])
    if location=='notes':
        data['notes']=42
    else:
        data['violations'][0]['description']=None
    bad['raw_output']=json.dumps(data)
    with pytest.raises(ValueError,match='schema'):
        validate_records([bad,record('opaque_test')],manifest())


def test_report_preserves_failed_gate_instead_of_silently_skipping(tmp_path):
    from types import SimpleNamespace
    from physcheck.report.aggregate import main
    raw=tmp_path/'raw';raw.mkdir()
    info=manifest()
    validation=validate_records([record('opaque_control',False),record('opaque_test')],info)
    (raw/'part_b_manifest.json').write_text(json.dumps(info))
    (raw/'judge_validated.status.json').write_text(json.dumps(validation))
    main(SimpleNamespace(results_dir=tmp_path),part='b')
    board=json.loads((tmp_path/'leaderboard/part_b_leaderboard.json').read_text())
    assert board['valid'] is False
    assert board['violation_rate_by_model'] is None
    assert (tmp_path/'leaderboard/part_b_leaderboard.md').exists()


def test_real_cli_gate_failure_writes_status_but_not_validated_output(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    info=manifest()
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(info))
    raw=tmp_path/'judgments.jsonl'
    raw.write_text('\n'.join(json.dumps(r) for r in [record('opaque_control',False),record('opaque_test')]))
    output=tmp_path/'validated.jsonl'
    repo=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(repo/'benchmark'),'PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'-m','physcheck','judge','--input-file',str(raw),
                           '--manifest',str(manifest_path),'--output-file',str(output)],
                          cwd=repo,env=env,capture_output=True,text=True)
    assert result.returncode==2
    assert not output.exists()
    assert json.loads(output.with_suffix('.status.json').read_text())['calibration']['valid'] is False


def test_real_cli_success_can_feed_a_production_report(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]
    raw=tmp_path/'raw';raw.mkdir()
    manifest_path=raw/'part_b_manifest.json';manifest_path.write_text(json.dumps(manifest()))
    source=raw/'judge_outputs.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in [record('opaque_control'),record('opaque_test',False)]))
    output=raw/'judge_validated.jsonl'
    env={**os.environ,'PYTHONPATH':str(repo/'benchmark'),'PHYSCHECK_RESULTS_DIR':str(tmp_path),
         'PHYSCHECK_ARTIFACTS_DIR':str(tmp_path/'artifacts'),'PHYSCHECK_MODELS_DIR':str(tmp_path/'models'),
         'PYTHONDONTWRITEBYTECODE':'1'}
    judge=subprocess.run([sys.executable,'-m','physcheck','judge','--input-file',str(source),
                          '--manifest',str(manifest_path),'--output-file',str(output)],
                         cwd=repo,env=env,capture_output=True,text=True)
    assert judge.returncode==0,judge.stderr
    assert len(output.read_text().splitlines())==2
    report=subprocess.run([sys.executable,'-m','physcheck','report','--part','b'],
                          cwd=repo,env=env,capture_output=True,text=True)
    assert report.returncode==0,report.stderr
    board=json.loads((tmp_path/'leaderboard/part_b_leaderboard.json').read_text())
    assert board['valid'] is True and board['violation_rate_by_model']['m']==0


def test_real_cli_preserves_an_existing_checkpoint_on_a_second_run(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]
    source=tmp_path/'judge_outputs.jsonl'
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(manifest()))
    output=tmp_path/'validated.jsonl'
    env={**os.environ,'PYTHONPATH':str(repo/'benchmark'),'PYTHONDONTWRITEBYTECODE':'1'}
    command=[sys.executable,'-m','physcheck','judge','--input-file',str(source),
             '--manifest',str(manifest_path),'--output-file',str(output)]
    source.write_text('\n'.join(json.dumps(r) for r in [record('opaque_control'),record('opaque_test',False)]))
    first=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    status=output.with_suffix('.status.json')
    original_output=output.read_bytes();original_status=status.read_bytes()
    source.write_text('\n'.join(json.dumps(r) for r in [record('opaque_control',False),record('opaque_test')]))
    second=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True)
    assert second.returncode!=0
    assert output.read_bytes()==original_output
    assert status.read_bytes()==original_status
