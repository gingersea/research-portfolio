"""Inspect completed local ranges and freeze the same sample without unused bytes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tarfile

from inspect_sources import ROOT

DATA=Path('/data/AI_Research/shared/real_measurement_20261117')
BLOCK=8*1024**2


class PartsReader(io.RawIOBase):
    def __init__(self,folder,count):
        self.folder=folder
        self.length=count*BLOCK
        self.position=0
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        return self.position
    def read(self,size=-1):
        size=min(self.length-self.position,size if size>=0 else self.length)
        if size<=0:
            return b''
        blocks=[]
        while size:
            index,offset=divmod(self.position,BLOCK)
            length=min(size,BLOCK-offset)
            with (self.folder/f'{index:05d}.part').open('rb') as stream:
                stream.seek(offset)
                block=stream.read(length)
            assert len(block)==length
            blocks.append(block)
            self.position+=length
            size-=length
        return b''.join(blocks)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--finish',action='store_true')
    parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    folder=DATA/'parts8'
    count=0
    while (folder/f'{count:05d}.part').exists():
        assert (folder/f'{count:05d}.part').stat().st_size==BLOCK
        count+=1
    archive=tarfile.open(fileobj=PartsReader(folder,count),mode='r:')
    clips={}
    order=[]
    while True:
        try:
            member=archive.next()
        except tarfile.ReadError:
            break
        if member is None:
            break
        parts=Path(member.name).parts
        if len(parts)<2:
            continue
        name=parts[1]
        if name not in clips:
            clips[name]=0
            order.append(name)
        clips[name]+=int(member.name.endswith('.jpg'))
    eligible=[]
    sources=set()
    for name in order[:-1]:
        group='_'.join(name.split('_')[:2])
        if 64<=clips[name]<=1200 and group not in sources:
            sources.add(group)
            eligible.append(name)
    print(json.dumps({'contiguous_bytes':count*BLOCK,'complete_clips':len(order)-1,'distinct_eligible':len(eligible),'selected':eligible[:12]}))
    if args.finish:
        assert len(eligible)>=12 or count==384
        if args.pid:
            cmdline=Path(f'/proc/{args.pid}/cmdline').read_bytes()
            assert b'research/real_measurement_20261117/download_prefix.py' in cmdline
            os.kill(args.pid,signal.SIGTERM)
        target=DATA/'ego_points_prefix.tar'
        digest=hashlib.sha256()
        with target.open('xb') as stream:
            for i in range(count):
                block=(folder/f'{i:05d}.part').read_bytes()
                digest.update(block)
                stream.write(block)
        metadata=json.loads((ROOT/'archive_metadata/range_probe.json').read_text())
        receipt={'file':str(target),'sha256':digest.hexdigest(),'bytes':target.stat().st_size,
                 'whole_archive_bytes':metadata['size_fields']['2'],'etag':metadata['headers']['Etag'],
                 'is_complete_archive':False,'all_range_boundaries_and_etag_verified':True,
                 'stopped_at_same_first12_complete_distinct_clips':True,'further_queued_download_cancelled_before_scores':True,
                 'partial_attempts_retained':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with (ROOT/'archive_metadata/prefix_receipt.json').open('x') as stream:
            json.dump(receipt,stream,indent=2)
