import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/bask/projects/d/duanj-ai-imaging/lxh/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from base import BaseAgent
from PreText import build_medical_documents
import numpy as np
import json
import random
import torch
import pandas as pd
import re
import csv
from func import process_all_folders

def set_seed(seed=0):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.set_float32_matmul_precision('high')
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def main():
    set_seed(42)
    # Root directory containing input document files (e.g., passport, ID card, driver license images)
    DOC_ROOT = "/bask/homes/z/zhangyzz/KB/doc/"
    
    # Root directory for storing extracted results and generated outputs
    RESULT_ROOT = "/bask/homes/z/zhangyzz/KB/result"


    system_prompt_keywords = ''
    user_prompt_keywords = (
                            "Read the text in the documents.\n"
                            "Do NOT repeat, transcribe, summarize, explain, infer, or generate any content."
                        )

    system_prompt_keywords2 = (
                                "You are an information extraction system.\n"
                                "Your task is to extract fields from the document.\n"
                                "Do NOT guess, infer, normalize, translate, or reformat any value.\n"
                                "Return ONE valid JSON object only."
                            )

    process_all_folders(DOC_ROOT,RESULT_ROOT,system_prompt_keywords,user_prompt_keywords,system_prompt_keywords2,max_attempts=5)

if __name__ == "__main__":
    main()
