from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_core.language_models import LLM
from pydantic import BaseModel, Field
from typing import Optional
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import hashlib
import numpy as np
# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
class InfeDeepSeekLLM(LLM, BaseModel):
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                                                            self.model_id, 
                                                            torch_dtype=torch.bfloat16,
                                                            trust_remote_code=True,
                                                            device_map="auto",
                                                            max_memory={0: "20GiB", 1: "20GiB"}
                                                            )
        print("DeepSeek model loaded successfully!")

    def _infe_optimized(self, system_prompt: str, user_prompt: str,item=[]) -> str:
        if hasattr(self.model, 'hf_device_map'):
            first_layer_device = list(self.model.hf_device_map.values())[0]
            model_device = torch.device(first_layer_device)
            print(f"[DEBUG] Auto device map detected, using: {model_device}")
        else:
            model_device = next(self.model.parameters()).device
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("{user_prompt}")
        ])

        full_prompt = prompt_template.format_messages(
            system_prompt=system_prompt, 
            user_prompt=user_prompt)
        
        prompt_text = "\n".join([msg.content for msg in full_prompt])

        inputs = self.tokenizer(prompt_text, 
                                # max_length=self.config.get("max_new_tokens", 4096),    
                                # padding="longest",    
                                # truncation=True,    
                                return_tensors="pt").to(model_device)

        # save text embedding
        # save_emb = True
        # save_dir = "./emb" 
        # if save_emb and save_dir:
        # os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            out_emb = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True)

        last_hidden = out_emb.hidden_states[-1]  # [B, T, H]
        attn = inputs.get("attention_mask", None)

        if attn is None:
            emb = last_hidden.mean(dim=1)  # [B, H]
        else:
            attn = attn.to(last_hidden.device).unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
            emb = (last_hidden * attn).sum(dim=1) / (attn.sum(dim=1) + 1e-12)  # [B, H]

        emb = emb[0].float().detach().cpu().numpy()  # [H]
            # key_str = system_prompt + "\n" + user_prompt
            # key = hashlib.md5(key_str.encode("utf-8")).hexdigest()
            # np.save(os.path.join(save_dir, f"text_{key}.npy"), emb)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 64),
                do_sample=self.config.get("do_sample", True),
                top_k=self.config.get("top_k", 40),
                top_p=self.config.get("top_p", 0.95),
                temperature=self.config.get("temperature", 0.8),
                repetition_penalty=self.config.get("repeat_penalty", 1.1),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response.split('\n</think>\n\n', 1)[-1].strip()
        return generated,emb
    
    def get_tokenizer_model(self) -> tuple:
        return self.tokenizer, self.model

    @property
    def _llm_type(self) -> str:
        return "infe_deepseek_optimized"

    def _call(self, prompt: str, stop=None) -> str:
        return self._infe_optimized("", prompt, [])
