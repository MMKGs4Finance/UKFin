from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_core.language_models import LLM
from pydantic import BaseModel, Field
from typing import Optional

class InfeGemma3LLM(LLM, BaseModel):
    model_id: str = "google/gemma-3-27b-it"
    config: dict = Field(default_factory=dict)
    
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    device: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        print("Gemma 3 LLM model loaded successfully")

    def _infe_optimized(self, system_prompt: str, user_prompt: str, item=[]) -> str:
        if hasattr(self.model, "hf_device_map"):
            first_layer_device = list(self.model.hf_device_map.values())[0]
            model_device = torch.device(first_layer_device)
        else:
            model_device = next(self.model.parameters()).device

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model_device)

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                do_sample=self.config.get("do_sample", True),
                top_k=self.config.get("top_k", 40),
                top_p=self.config.get("top_p", 0.95),
                temperature=self.config.get("temperature", 0.8),
                repetition_penalty=self.config.get("repeat_penalty", 1.1),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        generated_tokens = generation[0][inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded.strip(),_

    def get_tokenizer_model(self) -> tuple:
        return self.tokenizer, self.model

    @property
    def _llm_type(self) -> str:
        return "infe_Gemma3LLM"

    def _call(self, prompt: str, stop=None) -> str:
        return self._infe_optimized("", prompt, [])
