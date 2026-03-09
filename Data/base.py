from typing import Dict, Any
from PaliGemmaVLM import InfePaliGemmaVLM
from QwenVLM import InfeQwenVLM
from DeepSeekLLM import InfeDeepSeekLLM
from Gemma3LLM import InfeGemma3LLM
from enum import Enum

INFER_CLASS_MAP = {
    "InfeQwenVLM": InfeQwenVLM,
    "InfeDeepSeekLLM": InfeDeepSeekLLM,
    "InfePaliGemmaVLM": InfePaliGemmaVLM,
    "InfeGemma3LLM": InfeGemma3LLM}

class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    DECISION_MAKER = "Decision Maker"
    EXPERT_GATHERER = "Expert Gatherer"

class MedicalSpecialty(Enum):
    """Medical specialty enumeration for HF analysis."""
    CRITICAL_CARE = "Critical Care Medicine"
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    INFECTIOUS_DISEASE = "Infectious Disease"
    NEPHROLOGY = "Nephrology"
    HEMATOLOGY = "Hematology"
    ENDOCRINOLOGY = "Endocrinology"
    
class AgentRole(Enum):
    """Enumeration for different agent roles in MDAgents."""
    MODERATOR = "Moderator"
    RECRUITER = "Recruiter"
    GENERAL_DOCTOR = "General Doctor"
    SPECIALIST = "Specialist"
    TEAM_LEAD = "Team Lead"
    DECISION_MAKER = "Decision Maker" # For final decision synthesis
    

# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# deepseek-ai/DeepSeek-R1-0528
# 0.01 0.31 0.71 1.21 

LLM_MODELS_SETTINGS = {
     "InfeDeepSeekLLM": {
                        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  
                        "comment": "DeepSeek R1 Distilled Model",
                        
                        "reasoning": True,
                        "device": "auto",
                        "temperature": 0.6,  
                        "max_new_tokens": 128,
                        "top_k": 40,
                        "top_p": 1.0,
                        "typical_p": 1.0,
                        "do_sample": True,
                        # "num_beams": 4,  
                        "repeat_penalty": 1.1,
                        "max_length":4096
                    },
     
      "InfeQwenVLM": {
                        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",  
                        "comment": "QwenVLM",
                        
                        "reasoning": True,
                        "device": "auto",
                        "temperature": 0.01,    
                        "max_new_tokens": 256,
                        "top_k": 40,
                        "top_p": 1.0,
                        "typical_p": 1.0,
                        "do_sample": True,
                        "repeat_penalty": 1.1,
                        "num_beams": 4,  
                         
                        "max_length":4096
                    },

    # google/paligemma2-10b-mix-224
      "InfePaliGemmaVLM": {
                        "model_name": "google/paligemma2-10b-mix-448",
                        "comment": "Paligemma",
                        
                        "reasoning": True,
                        "device": "auto",
                        "temperature": 0.6,  
                        "max_new_tokens": 2048,
                        "top_k": 40,
                        "top_p": 1.0,
                        "typical_p": 1.0,
                        "do_sample": True,
                        "repeat_penalty": 1.2,
                        "num_beams": 1,  
                        "max_length":4096
                    },
      
      "InfeGemma3LLM": {
                        "model_name": "google/gemma-3-27b-it",  
                        "comment": "google medgemma text",
                        
                        "reasoning": True,
                        "device": "auto",
                        "temperature": 0.01,    
                        "max_new_tokens": 256,
                        "top_k": 40,
                        "top_p": 1.0,
                        "typical_p": 1.0,
                        "do_sample": True,
                        "repeat_penalty": 1.1,
                        # "num_beams": 4, 
                        
                        "max_length":4096 
                    },
      
    }

class BaseAgent:
    _shared_pipelines = {}
    
    def __init__(self,
                 agent_id: str,
                 agent_type: AgentType,       
                 model_key: str = "InfeDeepSeekLLM"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key

        self.config = LLM_MODELS_SETTINGS[model_key]
        self.pipeline = None
        self.model_name = None
        self._initialize_local_model()
        self.memory = []
        
    def _initialize_local_model(self):
        try:
            model_name = self.config["model_name"]

            if model_name in BaseAgent._shared_pipelines:
                self.pipeline = BaseAgent._shared_pipelines[model_name]
                self.model_name = model_name
                return
            
            infer_class = INFER_CLASS_MAP[self.model_key]
            self.pipeline = infer_class(model_id=model_name,config=self.config )

            BaseAgent._shared_pipelines[model_name] = self.pipeline
            self.model_name = model_name

        except Exception as e:
            print(f"load model error: {e}")
            raise e
        
    def call_llm(self,
             system_message: Dict[str, str],
             user_message: Dict[str, Any],
             item=[]) -> str:
        try:
            system_prompt = system_message
            user_prompt = user_message

            response,emb = self.pipeline._infe_optimized(system_prompt,user_prompt,item)
            return response, emb
        except Exception as e:
            raise Exception(f"Local LLM call failed: {e}")
    
    def close(self):
        if self.pipeline is not None:
            if hasattr(self.pipeline, "close"):
                self.pipeline.close()
            self.pipeline = None
        import torch
        torch.cuda.empty_cache()
    
    def clear_memory(self):
        """Clears the agent's conversation memory."""
        self.memory = []
        print(f"Cleared memory for agent {self.agent_id}")