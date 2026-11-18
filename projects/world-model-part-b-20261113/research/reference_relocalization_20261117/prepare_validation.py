"""Prepare frozen participant-disjoint clips; future labels stay separate."""
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import sys
import tarfile
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha

if __name__=='__main__':
    method=json.loads((ROOT/'method_freeze.json').read_text())
    for path,digest in method['sources'].items():assert sha(Path(path))==digest
    selection=json.loads((ROOT/'selection.json').read_text())
    archive_path=Path(selection['archive'])
    assert sha(archive_path)==selection['sha256']
    assert selection['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    chosen={r['id']:r for r in selection['selected']}
    output=ROOT/'data_validation';output.mkdir(exist_ok=False)
    archive=tarfile.open(archive_path,mode='r:')
    members={name:[] for name in chosen}
    while True:
        try: member=archive.next()
        except tarfile.ReadError:break
        if member is None:break
        parts=Path(member.name).parts
        if len(parts)>1 and parts[1] in chosen and member.isfile() and member.offset_data+member.size<=selection['bytes']:
            members[parts[1]].append(member)
    manifest={'utc_frozen_before_inference':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':sha(ROOT/'selection.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
        'selection':'archive-order first eligible OOV-return clip per unseen participant, up to6 within first3GiB',
        'excluded_prior_participants':selection['excluded_participants'],'cases':[]}
    for name,record in chosen.items():
        assert record['participant'] not in selection['excluded_participants']
        files=members[name]
        rgbs=sorted([m for m in files if '/rgbs/' in m.name and m.name.endswith('.jpg')],key=lambda m:m.name)
        assert len(rgbs)==record['frames']
        ann_members=[m for m in files if m.name.endswith('/annot.npz')];assert len(ann_members)==1
        raw=archive.extractfile(ann_members[0]).read();ann=np.load(io.BytesIO(raw),allow_pickle=False)
        assert ann['trajs_2d'].shape[0]==len(rgbs)
        assert (ann['valids'][0]==1).all() and (ann['visibs'][0]==1).all()
        assert not (ann['visibs'].astype(bool)&(ann['out_of_view'].astype(bool)|ann['occluded'].astype(bool))).any()
        folder=output/name;folder.mkdir()
        (folder/'original_annot.npz').write_bytes(raw)
        video=[];images=[]
        for i,member in enumerate(rgbs):
            assert int(Path(member.name).stem.split('_')[-1])==i+1
            content=archive.extractfile(member).read()
            image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR);assert image is not None
            if i==0:height,width=image.shape[:2]
            assert image.shape[:2]==(height,width)
            video.append(cv2.cvtColor(cv2.resize(image,(512,288),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB))
            import hashlib
            images.append({'tar_member':member.name,'tar_offset':member.offset_data,'bytes':member.size,'sha256':hashlib.sha256(content).hexdigest()})
        xy=ann['trajs_2d'].astype(np.float32)*np.array([512/width,288/height],np.float32)
        n=xy.shape[1];q=np.column_stack([np.zeros(n,np.float32),xy[0,:,1],xy[0,:,0]])
        times=np.flatnonzero(ann['vis_valids'][1:].any(axis=1))+1
        np.savez_compressed(folder/'input.npz',video=np.stack(video),queries_tyx=q,requested_times=times)
        np.savez_compressed(folder/'truth.npz',xy=xy,coordinate_valid=ann['valids'].astype(bool),
            visibility_valid=ann['vis_valids'].astype(bool),visible=ann['visibs'].astype(bool),
            out_of_view=ann['out_of_view'].astype(bool),occluded=ann['occluded'].astype(bool))
        save(folder/'source_images.json',images)
        manifest['cases'].append({'id':name,'participant':record['participant'],'source_video':'_'.join(name.split('_')[:2]),
            'frames':len(rgbs),'queries':n,'original_height':height,'original_width':width,
            'requested_times':times.tolist(),'source_return_events':record['return_events'],
            'files':{key:{'path':str(folder/file),'sha256':sha(folder/file)} for key,file in
                [('input','input.npz'),('truth','truth.npz'),('annotation','original_annot.npz'),('source_images','source_images.json')]}})
        print(json.dumps({'prepared':name,'frames':len(rgbs),'queries':n}),flush=True)
    archive.close();save(ROOT/'manifest_validation.json',manifest)
    files=[Path(__file__),ROOT/'run_native.py',ROOT/'run_descriptor.py',ROOT/'analyze.py',ROOT/'pairing_audit.py',
           PROJECT/'benchmark/physcheck/relations/descriptor.py',PROJECT/'benchmark/physcheck/relations/sparse.py',
           ROOT/'PROTOCOL.md',ROOT/'implementation_assets.json',ROOT/'checkpoint.json',
           PROJECT/'research/tracking_validation_20261115/assets.json']
    save(ROOT/'validation_freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'all_new_scores_unseen':True,
        'method_freeze_sha256':sha(ROOT/'method_freeze.json'),
        'manifest_sha256':sha(ROOT/'manifest_validation.json'),'sources':{str(p):sha(p) for p in files}})
    print(json.dumps({'cases':len(chosen),'participants':len({r['participant'] for r in chosen.values()}),'frozen':True}))
