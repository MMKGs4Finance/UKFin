import torch
from PIL import Image
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import LLM
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

class InfePaliGemmaVLM(LLM, BaseModel):
    model_id: str = "google/paligemma2-10b-mix-448"  
    config: dict = Field(default_factory=dict)

    model: Optional[object] = None
    processor: Optional[object] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._load_model()

    def _load_model(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "20GiB", 1: "20GiB"}
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("PaliGemma model loaded successfully!")

    def _infe_optimized(self, system_prompt: str, user_prompt: str, item=None) -> str:
        if item is None:
            item = {}

        if hasattr(self.model, "hf_device_map"):
            first_layer_device = list(self.model.hf_device_map.values())[0]
            model_device = torch.device(first_layer_device)
        else:
            model_device = next(self.model.parameters()).device

        image_paths = item.get("image_paths", []) if isinstance(item, dict) else []
        
        images = []
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))
       

        text = ("<image>\n" + system_prompt + "\n" + user_prompt + '.:').strip()

        


        inputs = self.processor(
            text=text,
            images=images if len(images) > 0 else None,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=self.config.get("do_sample", True),
                top_k=self.config.get("top_k", 40),
                top_p=self.config.get("top_p", 0.95),
                temperature=self.config.get("temperature", 0.8),
                repetition_penalty=self.config.get("repeat_penalty", 1.1),
                use_cache=True
            )

        # PaliGemma 直接 decode 全部即可
        response = self.processor.decode(generation[0], skip_special_tokens=True).strip()
        return response.split('.:', 1)[-1], None

    @property
    def _llm_type(self) -> str:
        return "infe_paligemma_optimized"

    def _call(self, prompt: str, stop=None) -> str:
        return self._infe_optimized("", prompt, {})
