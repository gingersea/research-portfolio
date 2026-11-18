"""Read-only checks of actual cached weights or the complete Part B run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models-dir',type=Path)
    parser.add_argument('--run',type=Path,default=STUDY/'run')
    args=parser.parse_args()
    if args.models_dir:
        checked=[]
        for filename in ['wan_weights_manifest.json','cog_weights_manifest.json']:
            manifest=json.loads((ROOT/'benchmark/results/raw'/filename).read_text())
            for record in manifest['files']:
                path=args.models_dir/manifest['repo']/record['file']
                assert path.stat().st_size==record['expected_bytes'],str(path)
                if record.get('expected_sha256'):
                    actual=digest(path)
                    assert actual==record['expected_sha256'],str(path)
                    checked.append({'repo':manifest['repo'],'revision':manifest['revision'],
                                    'file':record['file'],'sha256':actual})
        output={'passed':True,'verified_weight_files':len(checked),'weights':checked}
        (STUDY/'weights_verification.json').write_text(json.dumps(output,indent=2)+'\n')
        print(f'Weights verified against upstream recorded hashes: {len(checked)} files')
        return
    run=args.run.resolve();rows=[]
    for model in ['wan2.1_1.3b','cogvideox_5b']:
        rows.extend(json.loads((run/'private'/f'{model}.json').read_text()))
    prompts=json.loads((ROOT/'benchmark/physcheck/prompts/part_b.json').read_text())['prompts']
    controls=json.loads((ROOT/'benchmark/physcheck/prompts/calibration.json').read_text())['prompts']
    prompt_text={p['id']:p['text'] for p in prompts+controls}
    expected={(m,p['id'],s) for m in ['wan2.1_1.3b','cogvideox_5b'] for p in prompts for s in [0,1,2]}
    expected|={(m,p['id'],0) for m in ['wan2.1_1.3b','cogvideox_5b'] for p in controls}
    assert len(rows)==len(expected)==56
    assert {(r['model'],r['prompt_id'],r['seed']) for r in rows}==expected
    frame_count=0;sample_count=0
    native={'wan2.1_1.3b':(832,480,81,16.0),'cogvideox_5b':(720,480,49,8.0)}
    for record in rows:
        assert record['status']=='ok',record
        video=Path(record['video_path']);meta=json.loads(video.with_suffix('.json').read_text())
        assert digest(video)==meta['video_sha256']==record['video_sha256']
        assert meta['model']==record['model'] and meta['prompt_id']==record['prompt_id']
        assert meta['seed']==record['seed'] and meta['prompt']==prompt_text[record['prompt_id']]
        assert meta['num_inference_steps']==50 and meta['guidance_scale']==5.0
        width,height,expected_frames,expected_fps=native[record['model']]
        sample_dir=run/'blind/frames'/record['video_id']
        safe=json.loads((sample_dir/'payload.json').read_text())
        assert 'model' not in safe and 'expected_violation_kind' not in safe
        indices=[round(i*(expected_frames-1)/7) for i in range(8)]
        assert safe['video_id']==record['video_id']
        assert safe['sampled_frame_indices']==record['sampled_frame_indices']==indices
        assert safe['n_total_frames']==expected_frames
        assert len(list(sample_dir.glob('frame_*.png')))==8
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            assert (stream.width,stream.height)==(width,height)
            assert float(stream.average_rate)==meta['fps']==expected_fps
            frames=0
            for idx,frame in enumerate(container.decode(video=0)):
                frames+=1
                if idx in indices:
                    png=cv2.imread(str(sample_dir/f'frame_{idx:04d}.png'))
                    assert png is not None
                    rgb=cv2.cvtColor(png,cv2.COLOR_BGR2RGB)
                    assert np.array_equal(rgb,frame.to_ndarray(format='rgb24')),record['video_id']
                    sample_count+=1
        assert frames==record['decoded_frames']==meta['frames']==expected_frames
        frame_count+=frames
        assert (run/'judgments'/f'{record["video_id"]}.json').is_file()
        if not record['is_calibration']:
            assert (run/'calibration_gate.json').stat().st_mtime <= (run/'judgments'/f'{record["video_id"]}.json').stat().st_mtime
    assert sample_count==448
    output={'passed':True,'videos':56,'frames':frame_count,'calibration_videos':8,'production_videos':48,
            'independent_judgment_files':56,'sampled_frame_files':sample_count,
            'sampled_png_pixels_match_decoded_source':True,'native_settings_verified':True}
    (run/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
