"""Local fixed visual inference. No labels, external service or chat history."""
import json
import time


def parse_pc(text, video_id, frame_indices):
    start = text.find('{')
    if start < 0:
        raise ValueError('Missing JSON judgment')
    result, _ = json.JSONDecoder().raw_decode(text[start:])
    if result.get('video_id') != video_id or type(result.get('pc_score')) is not int or not 1 <= result['pc_score'] <= 5:
        raise ValueError('Unbound judgment or invalid PC score')
    if not isinstance(result.get('reason'), str) or not result['reason'].strip() or not isinstance(result.get('violations'), list):
        raise ValueError('Missing observation evidence')
    for event in result['violations']:
        if type(event.get('frame_idx')) is not int or event['frame_idx'] not in frame_indices:
            raise ValueError('Violation outside sampled source frames')
        if event.get('category') not in ['gravity', 'support', 'collision', 'fluid', 'rigid_body', 'other'] or not event.get('description'):
            raise ValueError('Invalid violation category/evidence')
    return result


class LocalVisualJudge:
    def __init__(self, directory, gpu=0):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        torch.set_num_threads(4)
        torch.manual_seed(20261117)
        torch.cuda.set_device(gpu)
        self.device = f'cuda:{gpu}'
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True,
                    min_pixels=256 * 28 * 28, max_pixels=1280 * 1280, use_fast=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(directory,
                    local_files_only=True, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', device_map={'': self.device}).eval()

    def generate(self, images, prompt):
        started = time.monotonic()
        messages = [{'role': 'user', 'content': [*({'type': 'image'} for _ in images), {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        output = self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                         skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output, {'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'family': 'Qwen', 'mode': 'local_transformers',
            'external_inference': False, 'independent_context_per_case': True, 'prior_assistant_context_supplied': False,
            'input_images': len(images), 'input_tokens': inputs['input_ids'].shape[1],
            'output_tokens': generated.shape[1] - inputs['input_ids'].shape[1],
            'max_new_tokens': 1400, 'do_sample': False, 'dtype': 'bfloat16', 'attention': 'sdpa',
            'elapsed_seconds': time.monotonic() - started}
