import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import hashlib
import requests
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from langchain_core.language_models import LLM
from typing import Optional
from pydantic import BaseModel, Field

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = (len(images) + num_columns - 1) // num_columns

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


class InfeQwenVLM(LLM, BaseModel):
    model_id: str
    config: dict = Field(default_factory=dict)
    total_pixels: int = 20480 * 28 * 28
    min_pixels: int = 16 * 28 * 28
    
    model: Optional[object] = None
    processor: Optional[object] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._load_model()
    
    def _load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("Qwen model loaded successfully!")
    
    def _infe_optimized(self, system_prompt,user_prompt: str,item=[]) -> str:
        if hasattr(self.model, 'hf_device_map'):
            first_layer_device = list(self.model.hf_device_map.values())[0]
            model_device = torch.device(first_layer_device)
            print(f"Auto device map detected, using: {model_device}")
        else:
            model_device = next(self.model.parameters()).device
            
        content_list = [{"type": "text", "text": user_prompt}]
        
        if isinstance(item, dict) and item.get("video_paths"):
            video_paths = item.get("video_paths")
        # video_paths = item.get("video_paths")
        # if len(video_paths) > 0:
            for vp in video_paths:
                content_list.append({
                    "video": vp, 
                    "total_pixels": self.total_pixels, 
                    "min_pixels": self.min_pixels
                })
        else:
            video_paths = []
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_list}]
        
        text = self.processor.apply_chat_template(
                                                messages, 
                                                tokenize=False, 
                                                # max_length=self.config.get("max_new_tokens", 4096),        
                                                # padding="longest",    
                                                # truncation=True,  
                                                add_generation_prompt=True)
        
        processor_inputs = {
            "text": [text],
            "padding": True,
            "return_tensors":"pt" }
        
        if len(video_paths) > 0:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages], return_video_kwargs=True
            )
            # fps_inputs = video_kwargs['fps']
            # print("video input:", video_inputs[0].shape)
            # num_frames, _, resized_height, resized_width = video_inputs[0].shape
            # print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
            
            processor_inputs["images"] = image_inputs
            processor_inputs["videos"] = video_inputs
            # processor_inputs["fps"] = fps_inputs

        inputs = self.processor(**processor_inputs)        
        # second_per_grid_ts: need to be a list
        if 'second_per_grid_ts' in inputs and isinstance(inputs['second_per_grid_ts'], torch.Tensor):
            inputs['second_per_grid_ts'] = inputs['second_per_grid_ts'].tolist()
        inputs = inputs.to(model_device)


        # save embedding
        # save_dir = "./emb" 
        # save_emb = True
        # if save_emb and save_dir:
        #     os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            out_emb = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        last_hidden = out_emb.hidden_states[-1]  # shape [B, T, H]
        attn = inputs.get("attention_mask", None)
    
        if attn is None:
            emb = last_hidden.mean(dim=1)  # [B, H]
        else:
            attn = attn.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
            emb = (last_hidden * attn).sum(dim=1) / (attn.sum(dim=1) + 1e-12)  # [B, H]
    
        emb = emb[0].float().detach().cpu().numpy()  # [H]
        
            # key_str = user_prompt + "|" + "|".join(video_paths) if len(video_paths) > 0 else user_prompt
            # key = hashlib.md5(key_str.encode("utf-8")).hexdigest()
            # np.save(os.path.join(save_dir, f"{key}.npy"), emb)

    

        output_ids = self.model.generate(**inputs,
                                        max_new_tokens=self.config.get("max_new_tokens", 512),
                                        do_sample=self.config.get("do_sample", True),
                                        # num_beams=self.config.get("num_beams", 4),
                                        top_k=self.config.get("top_k", 40),
                                        top_p=self.config.get("top_p", 0.95),
                                        typical_p=self.config.get("typical_p", 1.0),
                                        temperature=self.config.get("temperature", 0.8),
                                        repetition_penalty=self.config.get("repeat_penalty", 1.1),
                                        use_cache=True)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return output_text[0], emb

    
        
    

    @property
    def _llm_type(self) -> str:
        return "infe_qwen_optimized"

    def _call(self, prompt: str, stop=None) -> str:
        return self._infe_optimized("", prompt, [])










