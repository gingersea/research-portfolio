"""Fetch the published human test annotations, without downloading all videos."""
import json
from pathlib import Path
from huggingface_hub import HfApi
from fetch_ranges import transfer,sha

STUDY=Path(__file__).resolve().parent
TARGET=Path('/data/AI_Research/shared/physics_methods_20261114/data/videophy2_test')


def main():
    repo='videophysics/videophy2_test'
    info=HfApi(token=False).dataset_info(repo,files_metadata=True)
    rows=[]
    for item in info.siblings:
        if item.rfilename not in ['videophy2_test.csv','README.md']:continue
        path=transfer('datasets/'+repo,info.sha,item,TARGET,4)
        rows.append({'file':item.rfilename,'bytes':item.size,'sha256':sha(path),
                     'upstream_lfs_sha256':item.lfs.sha256 if item.lfs else None,'upstream_git_blob':item.blob_id})
    manifest={'repo':repo,'revision':info.sha,'local_dir':str(TARGET),'files':rows,
              'scope':'Published external human annotations; not newly labeled local generated videos.'}
    (STUDY/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'dataset':repo,'revision':info.sha,'verified_files':len(rows)},indent=2))


if __name__=='__main__':main()
