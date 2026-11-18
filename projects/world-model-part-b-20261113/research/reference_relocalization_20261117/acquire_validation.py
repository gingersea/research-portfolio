"""Bounded participant-disjoint return-stratum acquisition from the author tar."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
from urllib.parse import urljoin,urlparse

import numpy as np
import requests

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PRIOR=PROJECT/'research/real_measurement_20261117'
OLD=Path('/data/AI_Research/shared/real_measurement_20261117/parts8')
DATA=Path('/data/AI_Research/shared/reference_relocalization_20261117')
BLOCK=8*1024**2
LIMIT=384
HOSTS=['dropbox.com','dropboxusercontent.com']


def path_for(index):
    old=OLD/f'{index:05d}.part'
    return old if old.exists() else DATA/'parts8'/f'{index:05d}.part'


class Reader(io.RawIOBase):
    def __init__(self,count): self.length=count*BLOCK; self.position=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        return self.position
    def read(self,size=-1):
        remaining=min(max(0,self.length-self.position),size if size>=0 else self.length)
        result=[]
        while remaining:
            i,offset=divmod(self.position,BLOCK); n=min(remaining,BLOCK-offset)
            with path_for(i).open('rb') as stream: stream.seek(offset); block=stream.read(n)
            assert len(block)==n
            result.append(block); self.position+=n; remaining-=n
        return b''.join(result)


def inspect(excluded):
    count=0
    while count<LIMIT and path_for(count).exists(): count+=1
    archive=tarfile.open(fileobj=Reader(count),mode='r:')
    order=[]; records={}
    while True:
        try: member=archive.next()
        except tarfile.ReadError: break
        if member is None: break
        parts=Path(member.name).parts
        if len(parts)<2: continue
        name=parts[1]
        if name not in records:
            order.append(name); records[name]={'id':name,'frames':0,'return_events':0,'participant':name.split('_')[0]}
        row=records[name]
        row['frames']+=int(member.name.endswith('.jpg'))
        if member.name.endswith('/annot.npz') and member.offset_data+member.size<=count*BLOCK:
            annotations=np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False)
            known=annotations['vis_valids'].astype(bool)
            positive=known & annotations['visibs'].astype(bool) & annotations['valids'].astype(bool)
            out=known & annotations['out_of_view'].astype(bool)
            for point in range(known.shape[1]):
                pending=False
                for t in range(1,len(known)):
                    if out[t,point]: pending=True
                    elif pending and positive[t,point]: row['return_events']+=1; pending=False
            row['annotation_present']=True
    chosen=[]; participants=set()
    for name in order[:-1]:
        row=records[name]
        if row['participant'] not in excluded and row['participant'] not in participants and 64<=row['frames']<=1200 and row['return_events']>0 and row.get('annotation_present'):
            participants.add(row['participant']); chosen.append(row)
    return count,[records[n] for n in order[:-1]],chosen[:6],order[-1]


if __name__=='__main__':
    DATA.mkdir(exist_ok=True); folder=DATA/'parts8'; folder.mkdir(exist_ok=True)
    output=ROOT/'acquisition'; output.mkdir(exist_ok=False)
    old_manifest=json.loads((PRIOR/'manifest.json').read_text())
    excluded={r['participant'] for r in old_manifest['cases']}
    metadata=json.loads((PRIOR/'archive_metadata/range_probe.json').read_text())
    url=metadata['strings'][1].replace('dl=0','dl=1'); total=metadata['size_fields']['2']; tag=metadata['headers']['Etag']
    def fetch(index):
        start=index*BLOCK; end=start+BLOCK-1; errors=[]
        if path_for(index).exists(): return {'index':index,'status':'reused'}
        for attempt in range(6):
            try:
                session=requests.Session(); current=url+f'&range_start={start}&range_end={end}&study=reference20261117&retry={attempt}'
                for _ in range(8):
                    host=urlparse(current).hostname
                    assert any(host==h or host.endswith('.'+h) for h in HOSTS)
                    response=session.get(current,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,60),allow_redirects=False,stream=True)
                    if not response.is_redirect: break
                    current=urljoin(current,response.headers['Location']); response.close()
                assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}'
                assert response.headers.get('ETag')==tag
                temp=folder/f'{index:05d}.attempt{attempt}.partial'
                with temp.open('xb') as stream:
                    for chunk in response.iter_content(1<<18): stream.write(chunk)
                response.close(); assert temp.stat().st_size==BLOCK
                temp.replace(folder/f'{index:05d}.part')
                row={'index':index,'status':'ok','errors':errors}; break
            except Exception as error:
                errors.append(repr(error)); time.sleep(.5)
        else: row={'index':index,'status':'failed','errors':errors}
        (output/f'{index:05d}.json').write_text(json.dumps(row,indent=2)); return row
    start=time.monotonic()
    while True:
        count,scanned,chosen,boundary=inspect(excluded)
        print(json.dumps({'bytes':count*BLOCK,'scanned':len(scanned),'qualifying_new_participants':len(chosen),'ids':[r['id'] for r in chosen],'seconds':time.monotonic()-start}),flush=True)
        if len(chosen)>=6 or count==LIMIT: break
        indices=[i for i in range(count,min(count+12,LIMIT)) if not path_for(i).exists()]
        with ThreadPoolExecutor(12) as pool: results=list(pool.map(fetch,indices))
        if any(r['status']=='failed' for r in results):
            raise RuntimeError('Transport failed after bounded retries; original attempts retained')
    archive_path=DATA/'ego_points_validation_prefix.tar'; digest=hashlib.sha256()
    with archive_path.open('xb') as stream:
        for i in range(count):
            block=path_for(i).read_bytes(); digest.update(block); stream.write(block)
    record={'archive':str(archive_path),'bytes':count*BLOCK,'sha256':digest.hexdigest(),'whole_archive_bytes':total,'etag':tag,
        'excluded_participants':sorted(excluded),'selected':chosen,'scanned':scanned,'boundary_clip':boundary,
        'full_archive_claim':False,'no_new_prediction_scores_read':True,'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()}
    (ROOT/'selection.json').write_text(json.dumps(record,indent=2)); print(json.dumps({'selection_frozen':True,'new_participants':len(chosen)}))
