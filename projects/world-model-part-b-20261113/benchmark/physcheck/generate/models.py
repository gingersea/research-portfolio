"""Model registry for physcheck generation.

Each entry maps a model key to everything `pipeline.generate_video` needs to
load and run the model. Models are chosen for single-GPU (32 GB) feasibility
and permissive licensing.
"""

from __future__ import annotations

MODEL_REGISTRY: dict[str, dict] = {
    "wan2.1_1.3b": {
        "model_id": "wan2.1_1.3b",
        "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "pipeline_cls": "WanPipeline",
        "torch_dtype": "bfloat16",
        "width": 832,
        "height": 480,
        "num_frames": 81,
        "fps": 16.0,
        "license": "apache-2.0",
    },
    "cogvideox_5b": {
        "model_id": "cogvideox_5b",
        "hf_repo": "zai-org/CogVideoX-5b",
        "pipeline_cls": "CogVideoXPipeline",
        "torch_dtype": "bfloat16",
        "width": 720,
        "height": 480,
        "num_frames": 49,
        "fps": 8.0,
        "license": "other (CogVideoX license)",
    },
    # Fallback if Wan2.1 1.3B is insufficient
    "wan2.2_ti2v_5b": {
        "model_id": "wan2.2_ti2v_5b",
        "hf_repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pipeline_cls": "WanImageToVideoPipeline",
        "torch_dtype": "bfloat16",
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "fps": 16.0,
        "license": "apache-2.0",
    },
}

#: Models run in the production benchmark this month.
PRODUCTION_MODELS: list[str] = ["wan2.1_1.3b", "cogvideox_5b"]
