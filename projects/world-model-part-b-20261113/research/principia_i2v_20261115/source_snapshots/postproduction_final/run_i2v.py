"""Unmodified native LTX I2V inference with read-only conditioning receipts."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from diffusers import LTXImageToVideoPipeline

STUDY = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()


def save_video(pixels, directory, stem, fps):
    array_path = directory / f'{stem}_frames.npz'
    video_path = directory / f'{stem}.mp4'
    if array_path.exists() or video_path.exists():
        raise FileExistsError('Refuse to replace generated raw/video output')
    np.savez_compressed(array_path, frames=pixels)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-video_size', f'{pixels.shape[2]}x{pixels.shape[1]}', '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', str(video_path)], input=pixels.tobytes(), check=True)
    Image.fromarray(pixels[0]).save(directory / f'{stem}_actual_first.png')
    return {'array': str(array_path.relative_to(STUDY)), 'array_sha256': sha(array_path), 'video': str(video_path.relative_to(STUDY)), 'video_sha256': sha(video_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--initial', choices=['A', 'B'], required=True)
    parser.add_argument('--mode', choices=['smoke', 'production'], default='production')
    parser.add_argument('--run-name', default='primary')
    args = parser.parse_args()
    if not args.run_name.replace('_', '').isalnum():
        raise ValueError('Run name must be an identifier')
    torch.set_num_threads(8)
    manifest = json.loads((STUDY / 'native_assets_complete.json').read_text())
    assert Path(manifest['local_dir']).resolve() == args.model.resolve()
    for item in manifest['files']:
        assert sha(args.model / item['file']) == item['sha256'], item['file']
    assert json.loads((STUDY / 'instrument_preflight.json').read_text())['passed']
    setup = next(row for row in json.loads((STUDY / 'input_manifest.json').read_text()) if row['initial_id'] == args.initial)
    image_path = STUDY / setup['image']
    assert sha(image_path) == setup['image_sha256']
    image = Image.open(image_path).convert('RGB')
    smoke = args.mode == 'smoke'
    width, height, count, steps = (192, 128, 9, 1) if smoke else (768, 512, 97, 50)
    if smoke:
        image = image.resize((width, height))
    directory = STUDY / ('smoke' if smoke else 'generation') / args.run_name / args.initial
    directory.mkdir(parents=True, exist_ok=True)
    pipe = LTXImageToVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    native_prepare = pipe.prepare_latents
    signature = inspect.signature(native_prepare)
    conditioning_receipts = []

    def observe_prepare(*positional, **keywords):
        arguments = signature.bind(*positional, **keywords)
        arguments.apply_defaults()
        original_image = arguments.arguments['image']
        latents, mask = native_prepare(*positional, **keywords)
        unconditioned = latents[:, mask[0].eq(0)]
        conditioning_receipts.append({'image_tensor_shape': list(original_image.shape), 'image_tensor_sha256': tensor_sha(original_image), 'packed_latent_shape': list(latents.shape), 'conditioning_mask_shape': list(mask.shape), 'conditioned_tokens': int(mask.gt(0).sum().item()), 'unconditioned_noise_sha256': tensor_sha(unconditioned), 'native_prepare_source_sha256': sha(Path(inspect.getfile(type(pipe)))), 'observer_mutated_image_or_latents': False})
        return latents, mask

    pipe.prepare_latents = observe_prepare
    protocol = {'pipeline_class': type(pipe).__name__, 'model_revision': manifest['revision'], 'native_asset_manifest_sha256': sha(STUDY / 'native_assets_complete.json'), 'input_sha256': setup['image_sha256'], 'input_id': args.initial, 'height': height, 'width': width, 'frames': count, 'steps': steps, 'fps': 24, 'guidance_scale': 3., 'decode_timestep': 0., 'negative_prompt': 'worst quality, blurry, jittery, distorted', 'prompt': setup['prompt'], 'scheduler': dict(pipe.scheduler.config), 'pipeline_source_sha256': sha(Path(inspect.getfile(type(pipe)))), 'runner_sha256': sha(Path(__file__)), 'scope': 'Native image-condition path; original generated first frame retained, no pixel replacement. Read-only conditioning observer.'}
    protocol_path = directory / 'executed_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing run uses a different protocol')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    device = torch.device(f'cuda:{args.gpu}')
    for seed in ([42] if smoke else [42, 43]):
        stem = f'{args.initial}_seed{seed}'
        record_path = directory / f'{stem}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['status'] == 'ok'
            assert sha(STUDY / record['video']) == record['video_sha256']
            assert sha(STUDY / record['array']) == record['array_sha256']
            continue
        conditioning_receipts.clear()
        started = time.monotonic()
        print(json.dumps({'event': 'start', 'mode': args.mode, 'initial': args.initial, 'seed': seed}), flush=True)
        try:
            result = pipe(image=image, prompt=setup['prompt'], negative_prompt=protocol['negative_prompt'], width=width, height=height, num_frames=count, frame_rate=24, num_inference_steps=steps, guidance_scale=3., decode_timestep=0., generator=torch.Generator(device=device).manual_seed(seed), output_type='np').frames[0]
            assert len(conditioning_receipts) == 1
            assert conditioning_receipts[0]['conditioned_tokens'] == (width // 32) * (height // 32)
            assert result.shape == (count, height, width, 3) and np.isfinite(result).all()
            pixels = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
            files = save_video(pixels, directory, stem, 24)
            record = {'status': 'ok', 'initial_id': args.initial, 'seed': seed, **files, 'actual_first_frame_sha256': sha(directory / f'{stem}_actual_first.png'), 'frames': count, 'fps': 24, 'elapsed_seconds': time.monotonic() - started, 'conditioning': conditioning_receipts[0], 'first_frame_replaced': False, 'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device)}
        except Exception as error:
            record = {'status': 'api_failure', 'initial_id': args.initial, 'seed': seed, 'exception_type': type(error).__name__, 'exception': str(error), 'elapsed_seconds': time.monotonic() - started, 'conditioning_receipts': list(conditioning_receipts)}
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            raise
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        pipe.maybe_free_model_hooks()
        print(json.dumps({'event': 'complete', **record}), flush=True)


if __name__ == '__main__':
    main()
