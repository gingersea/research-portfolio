"""Verify source freeze, exclusion provenance, masks, inputs and all predictions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha,verify_file
from physcheck.relations.sparse import inside

if __name__=='__main__':
    selection=json.loads((ROOT/'selection.json').read_text())
    manifest=json.loads((ROOT/'manifest_validation.json').read_text())
    prior=json.loads((PROJECT/'research/real_measurement_20261117/manifest.json').read_text())
    prior_ids={c['participant'] for c in prior['cases']};new_ids={c['participant'] for c in manifest['cases']}
    assert set(selection['excluded_participants'])==prior_ids
    assert not(prior_ids&new_ids) and len(new_ids)==len(manifest['cases'])
    assert [c['id'] for c in manifest['cases']]==[c['id'] for c in selection['selected']]
    freeze=json.loads((ROOT/'validation_freeze.json').read_text())
    assert freeze['method_freeze_sha256']==sha(ROOT/'method_freeze.json')
    method=json.loads((ROOT/'method_freeze.json').read_text())
    for path,digest in method['sources'].items():assert sha(Path(path))==digest
    assert freeze['manifest_sha256']==sha(ROOT/'manifest_validation.json')
    for path,digest in freeze['sources'].items():assert sha(Path(path))==digest
    assert sha(Path(selection['archive']))==selection['sha256']
    old_receipt=json.loads((PROJECT/'research/real_measurement_20261117/archive_metadata/prefix_receipt.json').read_text())
    old_path=Path(old_receipt['file']);n=old_path.stat().st_size;digest=hashlib.sha256()
    with Path(selection['archive']).open('rb') as f:
        remaining=n
        while remaining:
            b=f.read(min(1<<20,remaining));assert b;digest.update(b);remaining-=len(b)
    assert digest.hexdigest()==old_receipt['sha256']
    records=[]
    for case in manifest['cases']:
        for record in case['files'].values():verify_file(record)
        data=np.load(verify_file(case['files']['input']),allow_pickle=False)
        g=np.load(verify_file(case['files']['truth']),allow_pickle=False)
        assert set(data.files)=={'video','queries_tyx','requested_times'}
        assert data['video'].shape==(case['frames'],288,512,3) and data['video'].dtype==np.uint8
        np.testing.assert_array_equal(data['queries_tyx'][:,[2,1]],g['xy'][0])
        assert (data['queries_tyx'][:,0]==0).all() and inside(data['queries_tyx'][:,[2,1]],(288,512)).all()
        expected=np.flatnonzero(g['visibility_valid'][1:].any(axis=1))+1
        np.testing.assert_array_equal(data['requested_times'],expected)
        negative=g['visibility_valid']&~g['visible'];negative[0]=False
        subtypes={'negative_oov':int((negative&g['out_of_view']).sum()),
                  'negative_occluded':int((negative&g['occluded']).sum()),
                  'negative_neither_subtype':int((negative&~g['out_of_view']&~g['occluded']).sum()),
                  'negative_both_subtypes':int((negative&g['out_of_view']&g['occluded']).sum())}
        row={'case':case['id'],'participant':case['participant'],'input_fields_verified':True,**subtypes,'predictions':{}}
        for kind,folder,key in [('native','predictions_native_validation','native'),('descriptor','predictions_descriptor_validation','raw')]:
            receipt=json.loads((ROOT/folder/(case['id']+'.json')).read_text())
            assert receipt['input_sha256']==case['files']['input']['sha256'] and receipt['future_truth_read'] is False
            assert receipt['manifest_sha256']==sha(ROOT/'manifest_validation.json')
            assert receipt['runner_sha256']==sha(ROOT/('run_native.py' if kind=='native' else 'run_descriptor.py'))
            if receipt['status']=='ok':
                path=ROOT/folder/(case['id']+'.npz');assert sha(path)==receipt['prediction_sha256']
                a=np.load(path,allow_pickle=False)
                assert not(a['reciprocal']&~a[key]).any()
                assert not(a[key]&~inside(a['xy'],(288,512))).any()
            row['predictions'][kind]=receipt['status']
        records.append(row)
    cached=np.load(ROOT/'matching_audit/real_descriptors.npz',allow_pickle=False)
    old=np.load(ROOT/'predictions_development/P01_103_start_3324_end_3632.npz',allow_pickle=False)
    np.testing.assert_array_equal(cached['xy'],old['xy'][308])
    save(ROOT/'integrity_receipt.json',{'participant_disjoint':True,'old_participants':sorted(prior_ids),'new_participants':sorted(new_ids),
        'frozen_sources_unchanged':True,'archive_prefix_sha256_verified':True,'reused_old_prefix_bytes_identical':True,
        'old_prefix_bytes':n,'label_input_isolation_fields_verified':True,'cases':records,'source_sha256':sha(Path(__file__)),
        'repeat_development_inference_xy_exact':True,'independent_family_review_pass':False,
        'checkpoint_publisher_checksum_claim':False,'physical_validity_claim':False})
    print(json.dumps({'cases':len(records),'participants':sorted(new_ids),'integrity_verified':True}))
