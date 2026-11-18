"""Exercise the real four-arm CLI on sparse labels and a planned model failure."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def dump(path,value):
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_model_retains_return_denominator_and_cached_replay_relocates(tmp_path):
    # Positive at2 follows known OOV at1; timestamp3 is wholly unannotated.
    case='P99_fixture'
    data=tmp_path/'copied_data'/case;data.mkdir(parents=True)
    truth=data/'truth.npz'
    np.savez_compressed(truth,xy=np.array([[[5,5]],[[0,0]],[[10,10]],[[0,0]]],float),
        coordinate_valid=np.array([[1],[0],[1],[0]],bool),visibility_valid=np.array([[1],[1],[1],[0]],bool),
        visible=np.array([[1],[0],[1],[0]],bool),out_of_view=np.array([[0],[1],[0],[0]],bool),
        occluded=np.zeros((4,1),bool))
    manifest=tmp_path/'manifest.json'
    dump(manifest,{'cases':[{'id':case,'participant':'P99','files':{
        'input':{'path':'/deliberately/unavailable/input.npz','sha256':'input-digest'},
        'truth':{'path':'/deliberately/unavailable/truth.npz','sha256':digest(truth)}}}]})
    native=tmp_path/'native';native.mkdir()
    descriptor=tmp_path/'descriptor';descriptor.mkdir()
    base={'input_sha256':'input-digest','future_truth_read':False}
    dump(native/(case+'.json'),dict(base,status='inference_failed',error='planned fixture failure'))
    pred=descriptor/(case+'.npz')
    np.savez_compressed(pred,xy=np.array([[[5,5]],[[5,5]],[[10,10]],[[50,50]]],float),
        raw=np.ones((4,1),bool),reciprocal=np.array([[1],[0],[1],[1]],bool))
    dump(descriptor/(case+'.json'),dict(base,status='ok',prediction_sha256=digest(pred)))
    output=tmp_path/'analysis'
    script=Path(__file__).resolve().parents[1]/'research/reference_relocalization_20261117/analyze.py'
    result=subprocess.run([sys.executable,str(script),'--manifest',str(manifest),'--native',str(native),
        '--descriptor',str(descriptor),'--output',str(output),'--split','new_participant_validation',
        '--relocated-data',str(data.parent)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    arms=json.loads((output/'summary.json').read_text())['arms']
    failed=arms['cotracker_native']['counts']
    assert (failed['positive'],failed['negative'],failed['returned_points'],failed['correct'])==(1,1,1,0)
    raw=arms['mast3r_raw']['counts']
    assert (raw['accepted'],raw['correct'],raw['false_visible'],raw['recovered'],raw['strict_recovered'])==(2,1,1,1,0)
    cycle=arms['mast3r_cycle']['counts']
    assert (cycle['accepted'],cycle['correct'],cycle['false_visible'],cycle['strict_recovered'])==(1,1,0,1)
    assert arms['mast3r_cycle']['participant_bootstrap']['precision']['ci95'] is None
