"""Versioned Part B production generation with calibration clips first.

One process per physical GPU/model. Private manifests and blinded judge payloads
are separate; no model identity or prompt is supplied to a judge.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import av

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmark'))
from physcheck.config import Config
from physcheck.generate.pipeline import generate_video, file_hash
from physcheck.judge.frames import sample_frames
from physcheck.judge.protocol import build_judge_payload


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['wan2.1_1.3b','cogvideox_5b'])
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    cfg=Config(ROOT,out/'private/artifacts',args.models_dir,out/'private/results',gpu_id=args.gpu)
    status_file=out/'private'/f'{args.model}.json'
    records=json.loads(status_file.read_text()) if status_file.exists() else []
    known={(r['prompt_id'],r['seed']):r for r in records}
    jobs=[]
    for key in ['calibration','part_b']:
        prompts=json.loads((ROOT/'benchmark/physcheck/prompts'/f'{key}.json').read_text())['prompts']
        for prompt in prompts:
            for seed in ([0] if key=='calibration' else [0,1,2]):
                jobs.append((key,prompt,seed))
    for key,prompt,seed in jobs:
        if (prompt['id'],seed) in known and known[(prompt['id'],seed)]['status']=='ok':
            old=known[(prompt['id'],seed)]
            assert file_hash(Path(old['video_path']))==old['video_sha256']
            continue
        vid='v_'+hashlib.sha256(f'part-b-frozen-20261113|{args.model}|{prompt["id"]}|{seed}'.encode()).hexdigest()[:20]
        record={'video_id':vid,'model':args.model,'prompt_id':prompt['id'],'seed':seed,
                'kind':prompt['kind'],'is_calibration':key=='calibration',
                'expected_violation_kind':prompt.get('expected_violation_kind')}
        start=time.monotonic()
        try:
            generated=generate_video(args.model,prompt['text'],seed,cfg,prompt_id=prompt['id'])
            video=Path(generated['video_path'])
            with av.open(str(video)) as container:
                stream=container.streams.video[0]
                fps=float(stream.average_rate)
                count=sum(1 for _ in container.decode(video=0))
            assert count==generated['metadata']['frames'] and fps==generated['metadata']['fps']
            # Config redirects frame output into a separate, opaque-ID namespace.
            blind_cfg=Config(ROOT,out/'blind',args.models_dir,out/'private/results',gpu_id=args.gpu)
            sampled=sample_frames(str(video),8,blind_cfg,vid)
            payload=build_judge_payload({'video_id':vid,'grid_path':sampled['grid_path'],'n_frames':8})
            safe={'video_id':vid,'prompt':payload['prompt'],'grid_path':sampled['grid_path'],
                  'sampled_frame_indices':sampled['indices'],'n_total_frames':count}
            write(out/'blind/frames'/vid/'payload.json',safe)
            record.update(status='ok',video_path=str(video),video_sha256=file_hash(video),
                          fps=fps,decoded_frames=count,sampled_frame_indices=sampled['indices'],
                          grid_path=sampled['grid_path'],generation_metadata=generated['metadata'])
        except Exception as error:
            record.update(status='error',error=f'{type(error).__name__}: {error}')
        record['elapsed_seconds']=time.monotonic()-start
        known[(prompt['id'],seed)]=record
        records=list(known.values())
        write(status_file,records)
        print(json.dumps({k:record[k] for k in ['video_id','prompt_id','seed','status','elapsed_seconds']}),flush=True)
    write(out/'private'/f'{args.model}_completed.json',{'expected':28,'ok':sum(r['status']=='ok' for r in records),'records':len(records)})


if __name__=='__main__':main()
