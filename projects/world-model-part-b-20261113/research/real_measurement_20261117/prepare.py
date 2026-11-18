"""Freeze a bounded archive-prefix sample and isolate inference inputs from truth."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys
import tarfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha


def main():
    receipt=json.loads((ROOT/'archive_metadata/prefix_receipt.json').read_text())
    source=Path(receipt['file'])
    assert sha(source)==receipt['sha256']
    archive=tarfile.open(source,mode='r:')
    clips={}
    order=[]
    boundary_error=None
    while True:
        try:
            member=archive.next()
        except tarfile.ReadError as error:
            boundary_error=str(error)
            break
        if member is None:
            break
        parts=Path(member.name).parts
        if len(parts)<2:
            continue
        name=parts[1]
        assert re.fullmatch(r'P\d+_\d+_start_\d+_end_\d+',name),name
        if name not in clips:
            clips[name]=[]
            order.append(name)
        if member.isfile() and member.offset_data+member.size<=source.stat().st_size:
            clips[name].append(member)
    # Prefix does not reach archive EOF. Last directory cannot be certified complete.
    complete=order[:-1]
    selected=[]
    groups=set()
    index=[]
    for name in complete:
        rgbs=sorted([m for m in clips[name] if '/rgbs/' in m.name and m.name.endswith('.jpg')],key=lambda m:m.name)
        annotations=[m for m in clips[name] if m.name.endswith('/annot.npz')]
        group='_'.join(name.split('_')[:2])
        eligible=64<=len(rgbs)<=1200 and len(annotations)==1
        choose=eligible and group not in groups and len(selected)<12
        index.append({'case':name,'source_video':group,'frames':len(rgbs),'eligible':eligible,'selected':choose})
        if choose:
            selected.append((name,group,rgbs,annotations[0]))
            groups.add(group)
    assert selected,'No complete eligible clips in frozen prefix'
    output=ROOT/'data'
    output.mkdir(exist_ok=False)
    manifest={'utc_frozen_before_inference':datetime.now(timezone.utc).isoformat(),
              'selection':'first12 eligible distinct-source clips in first3GiB author archive prefix; no outcome selection',
              'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'amendment_sha256':sha(ROOT/'PROTOCOL_v2.md'),
              'acquisition_stop_sha256':sha(ROOT/'PROTOCOL_v3.md'),
              'archive_prefix_sha256':receipt['sha256'],'complete_scanned':index,'boundary_clip':order[-1],
              'boundary_error':boundary_error,'cases':[]}
    for name,group,rgbs,ann_member in selected:
        folder=output/name
        folder.mkdir()
        ann_bytes=archive.extractfile(ann_member).read()
        (folder/'original_annot.npz').write_bytes(ann_bytes)
        ann=np.load(io.BytesIO(ann_bytes),allow_pickle=False)
        assert ann['trajs_2d'].shape[0]==len(rgbs)
        assert (ann['valids'][0]==1).all() and (ann['visibs'][0]==1).all()
        video=[]
        images=[]
        raw_folder=Path('/data/AI_Research/shared/real_measurement_20261117/extracted')/name
        raw_folder.mkdir(parents=True,exist_ok=False)
        for i,member in enumerate(rgbs):
            assert int(Path(member.name).stem.split('_')[-1])==i+1
            content=archive.extractfile(member).read()
            path=raw_folder/Path(member.name).name
            path.write_bytes(content)
            image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
            assert image is not None
            if i==0:
                height,width=image.shape[:2]
            assert image.shape[:2]==(height,width)
            video.append(cv2.cvtColor(cv2.resize(image,(512,288),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB))
            images.append({'file':str(path),'sha256':sha(path),'tar_offset':member.offset_data,'bytes':member.size})
        xy=ann['trajs_2d'].astype(np.float32)*np.array([512/width,288/height],np.float32)
        n=xy.shape[1]
        queries=np.column_stack([np.zeros(n,np.float32),xy[0,:,1],xy[0,:,0]])
        requested=np.flatnonzero(ann['vis_valids'][1:].any(axis=1))+1
        np.savez_compressed(folder/'input.npz',video=np.stack(video),queries_tyx=queries,requested_times=requested)
        np.savez_compressed(folder/'truth.npz',xy=xy,coordinate_valid=ann['valids'].astype(bool),
             visibility_valid=ann['vis_valids'].astype(bool),visible=ann['visibs'].astype(bool),
             out_of_view=ann['out_of_view'].astype(bool),occluded=ann['occluded'].astype(bool))
        save(folder/'source_images.json',images)
        manifest['cases'].append({'id':name,'source_video':group,'participant':group.split('_')[0],
             'frames':len(rgbs),'queries':n,'original_height':height,'original_width':width,
             'annotation_tar_offset':ann_member.offset_data,'requested_times':requested.tolist(),
             'files':{key:{'path':str(folder/file),'sha256':sha(folder/file)} for key,file in
                      [('input','input.npz'),('truth','truth.npz'),('annotation','original_annot.npz'),('source_images','source_images.json')]}})
    archive.close()
    save(ROOT/'manifest.json',manifest)
    save(ROOT/'pre_inference_source.json',{'utc':datetime.now(timezone.utc).isoformat(),
         'manifest_sha256':sha(ROOT/'manifest.json'),'all_new_scores_unseen':True,
         'source_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in
              [Path(__file__),ROOT/'run_models.py',ROOT/'analyze.py',PROJECT/'benchmark/physcheck/relations/sparse.py',
               PROJECT/'tests/test_sparse_measurement.py']}})
    print(json.dumps({'selected':len(selected),'participants':len({c['participant'] for c in manifest['cases']}),
                      'total_frames':sum(c['frames'] for c in manifest['cases']),'queries':sum(c['queries'] for c in manifest['cases'])}))


if __name__=='__main__':
    main()
