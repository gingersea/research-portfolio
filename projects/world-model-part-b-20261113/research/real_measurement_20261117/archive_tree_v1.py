"""Inspect archive listing before deciding what data to acquire."""
import json
import re
from inspect_sources import ROOT, get

if __name__=='__main__':
    raw=(ROOT/'archive_metadata/prefetch_1.bin').read_bytes()
    urls=[s.decode() for s in re.findall(rb'https://[^\x00-\x20]+?/p\.json',raw)]
    matches=[u for u in urls if '/archive_tree/' in u]
    for i,url in enumerate(matches):
        content,final,_=get(url,max_bytes=50_000_000)
        path=ROOT/'archive_metadata'/f'archive_tree_{i}.json'
        path.write_bytes(content)
        parsed=json.loads(content)
        print(json.dumps({'path':str(path),'type':str(type(parsed)),'top':str(parsed)[:1800]}))
