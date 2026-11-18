"""Paired inference using pinned, unchanged official VPT inference definitions.

This entry point excludes training-only imports. It is an adapted released-weight
inference trial, not a reproduction of training or cross-step auto-guidance.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline
from diffusers import WanTransformer3DModel as OriginalWanTransformer3DModel
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.transformers.transformer_wan import WanImageEmbedding, WanTimeTextImageEmbedding
from safetensors.torch import load_file

STUDY = Path(__file__).resolve().parent
SCRATCH = Path('/data/AI_Research/shared/physics_methods_20261114')
SOURCE = SCRATCH / 'source/vpt'
BASE = Path('/home/room115/jhy/world-model-physical-consistency/.worktrees/video-bench/benchmark/physcheck/models_cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
WEIGHTS = SCRATCH / 'models/vpt_ranged/pytorch_lora_weights.safetensors'
PROMPTS = {
    'bounce': 'A red rubber ball is released from rest above a flat floor. Static camera, full ball and floor visible.',
    'collision': 'A blue billiard ball rolls into a stationary red billiard ball on a level table. Static camera, both balls visible.',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def official_definitions():
    """Verify Git blobs before executing only the original inference AST nodes."""
    manifest = json.loads((SOURCE / 'SOURCE_MANIFEST.json').read_text())
    expected = {r['path']: r['git_blob'] for r in manifest['files']}
    for name in ['finetrainers/models/wan/base_specification.py', 'finetrainers/models/wan/custome.py']:
        data = (SOURCE / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected[name]
    source = SOURCE / 'finetrainers/models/wan/base_specification.py'
    tree = ast.parse(source.read_text())
    names = {'WanTripleDiscreteTimeTextImageEmbedding', '_maybe_enable_triple_discrete_timestep_conditioning', 'DualStreamTransformer3DModel'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    for package in ['finetrainers', 'finetrainers.models', 'finetrainers.models.wan']:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    name = 'finetrainers.models.wan.base_specification'
    module = types.ModuleType(name)
    module.__dict__.update({k: globals()[k] for k in ['torch', 'nn', 'math', 'Any', 'Dict', 'List', 'Optional', 'Tuple', 'Union', 'OriginalWanTransformer3DModel', 'TimestepEmbedding', 'WanImageEmbedding', 'WanTimeTextImageEmbedding']})
    sys.modules[name] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), module.__dict__)
    spec = importlib.util.spec_from_file_location('finetrainers.models.wan.custome', SOURCE / 'finetrainers/models/wan/custome.py')
    custom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom)
    negatives = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'negative_prompt' for t in n.targets) and isinstance(n.value, ast.Constant)]
    assert len(negatives) == 1
    provenance = {'repo': manifest['repo'], 'revision': manifest['revision'], 'source_blobs': {k: expected[k] for k in expected if k.endswith(('base_specification.py', 'custome.py'))}, 'selected_ast_sha256': hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()).hexdigest()}
    return module, custom.CustomWanPipeline, negatives[0], provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['base', 'vpt'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    definitions, custom_cls, negative, provenance = official_definitions()
    assert digest(WEIGHTS) == 'fe9b57aded3514f819627d0fb25f4702f75fa0e28d74ac2bc5521fdc15fcd848'
    out = STUDY / ('smoke' if args.smoke else 'generation') / args.arm
    out.mkdir(parents=True, exist_ok=True)
    height, width, frames, steps = (128, 128, 5, 1) if args.smoke else (480, 832, 81, 50)
    pipe = WanPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE, subfolder='scheduler', shift=5.0, local_files_only=True)
    base_config = dict(pipe.transformer.config)
    if args.arm == 'vpt':
        # The full released patch/projection weights replace these original layers.
        # All intermediate pretrained blocks retain their original parameters.
        pipe.transformer.__class__ = definitions.DualStreamTransformer3DModel
        pipe.transformer.register_to_config(original_in_channels=16, original_out_channels=16, flow_in_channels=16, flow_out_channels=16, seg_in_channels=16, seg_out_channels=16, in_channels=48, out_channels=48)
        wrapper = custom_cls(pipe)
        wrapper.load_lora_weights(str(WEIGHTS))
        state = load_file(WEIGHTS)
        for key, value in state.items():
            if 'lora' not in key:
                actual = pipe.transformer.state_dict()[key.removeprefix('transformer.')]
                assert torch.equal(actual.cpu(), value.to(actual.dtype)), key
        assert pipe.transformer.patch_embedding.in_channels == 48
        assert pipe.transformer.proj_out.out_features == 192
    else:
        wrapper = pipe
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    noise = torch.randn((1, 16, (frames - 1) // 4 + 1, height // 8, width // 8), generator=torch.Generator().manual_seed(42), dtype=torch.float32).to(torch.bfloat16)
    noise_sha = hashlib.sha256(noise.view(torch.uint16).numpy().tobytes()).hexdigest()
    original_prepare = pipe.prepare_latents
    calls = 0

    def shared_rgb_noise(batch, channels, h, w, n, dtype, dev, generator, latents=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert batch == 1 and channels == 16 and (h, w, n) == (height, width, frames)
            return noise.to(device=dev, dtype=dtype).clone()
        return original_prepare(batch, channels, h, w, n, dtype, dev, generator, latents)

    pipe.prepare_latents = shared_rgb_noise
    metadata = {'arm': args.arm, 'prompts': PROMPTS, 'seed': 42, 'flow_seed': 44, 'seg_seed': 45, 'height': height, 'width': width, 'frames': frames, 'steps': steps, 'fps': 16, 'scheduler': dict(pipe.scheduler.config), 'rgb_noise_bf16_sha256': noise_sha, 'source': provenance, 'base_transformer_config': base_config, 'adapter_sha256': digest(WEIGHTS), 'negative_prompt_sha256': hashlib.sha256(negative.encode()).hexdigest(), 'scope': 'Released-weight adapted inference; no training, no cross-step auto-guidance; unlabeled outputs.'}
    (out / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    prompts = list(PROMPTS.items())[:1] if args.smoke else list(PROMPTS.items())
    for prompt_id, prompt in prompts:
        for scale in ([2.2] if args.arm == 'vpt' else [2.2, 3.2]):
            stem = f'{prompt_id}_seed42_gs{scale}'
            video_path = out / f'{stem}.mp4'
            record_path = out / f'{stem}.json'
            if video_path.exists() and record_path.exists():
                assert json.loads(record_path.read_text())['sha256'] == digest(video_path)
                continue
            calls = 0
            started = time.monotonic()
            extra = {'flow_guidance_scale': 0.0, 'seg_guidance_scale': 0.0, 'enable_inner_guidance': True} if args.arm == 'vpt' else {}
            print(json.dumps({'event': 'start', 'arm': args.arm, 'video': stem}), flush=True)
            # Official trainer validation runs mixed precision; the newly rebuilt
            # time-conditioning layers retain FP32 parameters in the release code.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = wrapper(prompt=prompt, negative_prompt=negative, height=height, width=width, num_frames=frames, num_inference_steps=steps, guidance_scale=scale, generator=torch.Generator(device=device).manual_seed(42), **extra).frames[0]
            assert len(result) == frames
            pixels = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
            raw_path = out / f'{stem}_frames.npz'
            np.savez_compressed(raw_path, frames=pixels)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '16', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(video_path)], input=pixels.tobytes(), check=True)
            record = {'video': str(video_path), 'sha256': digest(video_path), 'prompt': prompt, 'guidance_scale_parameter': scale, 'effective_text_cfg': scale + 1 if args.arm == 'vpt' else scale, 'elapsed_seconds': time.monotonic() - started, 'initial_rgb_noise_sha256': noise_sha, 'prepare_latents_calls': calls, 'frames_before_encoding': len(result), 'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device)}
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            pipe.maybe_free_model_hooks()
            print(json.dumps({'event': 'complete', **record}), flush=True)


if __name__ == '__main__':
    main()
