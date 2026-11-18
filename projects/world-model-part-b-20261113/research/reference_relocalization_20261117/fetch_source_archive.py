"""Acquire the pinned public implementation archive, no git checkout changes."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import requests

ROOT=Path(__file__).resolve().parent
ASSETS=Path('/data/AI_Research/shared/reference_relocalization_20261117')

if __name__=='__main__':
    ASSETS.mkdir(exist_ok=True)
    rev=json.loads((ROOT/'sources/revision.json').read_text())['sha']
    session=requests.Session()
    session.trust_env=False
    url=f'https://codeload.github.com/rhodriguerrier/PointSt3R/zip/{rev}'
    response=session.get(url,timeout=(15,120),allow_redirects=False)
    response.raise_for_status()
    assert not response.is_redirect
    raw=response.content
    path=ASSETS/'PointSt3R_source.zip'
    with path.open('xb') as stream:
        stream.write(raw)
    target=ASSETS/'source'
    target.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            parts=Path(name).parts[1:]
            if not parts:
                continue
            destination=target.joinpath(*parts)
            assert destination.resolve().is_relative_to(target.resolve())
            if name.endswith('/'):
                destination.mkdir(parents=True,exist_ok=True)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as stream:
                    stream.write(archive.read(name))
    record={'source_directory':str(target),'archive':str(path),'revision':rev,'url':url,
            'archive_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
            'files':[{'file':str(p.relative_to(target)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in target.rglob('*') if p.is_file()]}
    with (ROOT/'implementation_assets.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps({'directory':str(target),'bytes':len(raw),'files':len(record['files'])}))
