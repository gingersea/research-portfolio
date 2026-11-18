"""Single-image native VAE-mode reconstruction, separate from generated videos."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLLTXVideo

STUDY = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    output = STUDY / 'vae_roundtrip'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    vae = AutoencoderKLLTXVideo.from_pretrained(args.model, subfolder='vae', torch_dtype=torch.bfloat16, local_files_only=True).to(f'cuda:{args.gpu}')
    vae.enable_tiling()
    records = []
    with torch.inference_mode():
        for setup in json.loads((STUDY / 'input_manifest.json').read_text()):
            image = np.array(Image.open(STUDY / setup['image']).convert('RGB'))
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None, :, None].to(device=vae.device, dtype=vae.dtype) / 127.5 - 1
            latent = vae.encode(tensor).latent_dist.mode()
            decoded = vae.decode(latent, timestep=torch.zeros(1, device=vae.device, dtype=vae.dtype)).sample
            assert decoded.shape == tensor.shape and torch.isfinite(decoded).all()
            pixels = ((decoded[0, :, 0].float().permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
            pixels = np.rint(pixels * 255).astype(np.uint8)
            path = output / f'{setup["initial_id"]}_vae_mode.png'
            Image.fromarray(pixels).save(path)
            records.append({'initial_id': setup['initial_id'], 'input_sha256': setup['image_sha256'], 'reconstruction': str(path.relative_to(STUDY)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'mean_absolute_pixel_error': float(np.abs(pixels.astype(float) - image).mean()), 'latent_shape': list(latent.shape)})
    (output / 'receipt.json').write_text(json.dumps({'scope': 'Single-image VAE posterior mode at decode_timestep0; not a video or an assertion of exact production first-frame reconstruction.', 'records': records}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
