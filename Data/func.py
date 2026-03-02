from utils import preprocess_response_string,json_to_Passport,json_to_DriverLicence,json_to_Person,json_to_BirthCertificate,json_to_Income
import json
import os
from base import BaseAgent
from PreText import build_medical_documents

class FinancialTwin(BaseAgent):
    def analyze(self, system_message: str, user_message: str,item=[]):
        result, emb = self.call_llm(system_message = system_message,user_message=user_message,item=item)
        return str(result), emb

# ImageEmbedTwin = FinancialTwin("ImageEmbed", "Decision", "InfeQwenVLM")
ImageOcrTwin = FinancialTwin("ImageOCR", "Decision", "InfePaliGemmaVLM")

# TextEmbedTwin = FinancialTwin("TextEmbed", "Decision", "InfeDeepSeekLLM")
TextKeywordTwin = FinancialTwin("TextKeyword", "Decision", "InfeDeepSeekLLM")

def OCR_extract(system_prompt_keywords2, user_prompt_keywords2, item, required_keys, max_attempts=5):
    default = {
        "first_name": None,
        "last_name": None,
        "id_number": None
    }

    for attempt in range(max_attempts):
        try:
            result, _ = TextKeywordTwin.analyze(
                system_message=system_prompt_keywords2,
                user_message=user_prompt_keywords2,
                item=item
            )

            temp = preprocess_response_string(result)
            parsed_result = json.loads(temp)

            # 1. 必须是 dict
            if not isinstance(parsed_result, dict):
                print(f"Attempt {attempt+1} result not dict. Retrying...")
                continue

            # 2. 必须包含所有字段
            # required_keys = ["first_name", "last_name", "id_number"]
            if not all(k in parsed_result for k in required_keys):
                print(f"Attempt {attempt+1} missing keys. Retrying...")
                continue

            # 4. id_number 必须存在且为纯数字字符串
            if "id_number" in required_keys:
                id_number = parsed_result.get("id_number")
            
                if id_number is None:
                    print(f"Attempt {attempt+1} id_number is null. Retrying...")
                    continue
            
                if not isinstance(id_number, str) or not id_number.isdigit():
                    print(f"Attempt {attempt+1} invalid id_number. Retrying...")
                    continue

            # 5. 全部通过，返回
            return parsed_result, temp

        except Exception as e:
            print(f"Attempt {attempt+1} failed with exception: {e}. Retrying...")
            
    print("All attempts failed. Returning default.")
    return default, json.dumps(default)




    
def process_all_folders(DOC_ROOT,RESULT_ROOT,system_prompt_keywords,user_prompt_keywords,system_prompt_keywords2, max_attempts=5):
    os.makedirs(RESULT_ROOT, exist_ok=True)

    SAVE_FUNCS = {
        "Passport": json_to_Passport,
        "DriverLicence": json_to_DriverLicence,
        "Person": json_to_Person,
        "BirthCertificate": json_to_BirthCertificate,
    }

    for folder_name in sorted(os.listdir(DOC_ROOT)):
        folder_path = os.path.join(DOC_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(RESULT_ROOT, f"{folder_name}.csv")
        print("Processing folder", folder_path)
        print("Saving to", csv_path)

        for root, _, files in os.walk(folder_path):
            for fname in sorted(files):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    
                    img_path = os.path.join(root, fname)
    
                    item = {
                            "image_paths": [img_path],
                            "video_paths": [img_path]
                        }

                    # print("=== Image Embed ===")
                    # user_message = ''
                    # out_image_embed = ImageEmbedTwin.analyze(
                    #     system_message=system_message_embedding,
                    #     user_message=user_message,
                    #     item = item
                    # )
                    # print("=== Image Embed Out ===")
                    # print(out_image_embed)

                    # print("=== Image OCR ===")
                    out_image_Ocr,_ = ImageOcrTwin.analyze(
                                        system_message=system_prompt_keywords,
                                        user_message=user_prompt_keywords,
                                        item = item
                                    )
    
                    if folder_name == 'Passport':
                        user_prompt_keywords_format2 = (
                                "Return ONE valid JSON object with exactly the following keys:\n"
                                "\"first_name\": string or null,\n"
                                "\"last_name\": string or null,\n"
                                "\"id_number\": string or null (digits only, exactly 9 digits, no spaces, match regex ^[0-9]{9}$).\n\n"
        
                                "Rules:\n"
                                "1. Use the value only if it is clearly shown in the document.\n"
                                "2. If a field is missing or unclear, keep its value as null.\n"
                                "3. Do NOT guess, infer, normalize, or reformat values.\n"
                                "4. Output ONLY the JSON object. No explanations, no extra text."
                            )
                        required_keys = ["first_name", "last_name", "id_number"]
                        
                    elif folder_name == 'DriverLicence':
                        user_prompt_keywords_format2 = (
                                "Return ONE valid JSON object with exactly the following keys:\n"
                                "\"first_name\": string or null,\n"
                                "\"last_name\": string or null,\n"
                                "\"birthDate\": string or null (ONLY if clearly shown in format DD/MM/YYYY),\n"
                                "\"issueDate\": string or null (ONLY if clearly shown in format DD/MM/YYYY),\n"
                                "\"expiryDate\": string or null (ONLY if clearly shown in format DD/MM/YYYY).\n\n"
    
                                "Rules:\n"
                                "1. Use the value only if it is clearly shown in the document.\n"
                                "2. If a field is missing or unclear, keep its value as null.\n"
                                "3. Do NOT guess, infer, normalize, or reformat values.\n"
                                "4. Output ONLY the JSON object. No explanations, no extra text."
                            )
                        required_keys = ["first_name", "last_name","birthDate", "issueDate","expiryDate"]
    
                    elif folder_name == 'Person':	
                        user_prompt_keywords_format2 = (
                                "Return ONE valid JSON object with exactly the following keys:\n"
                                "\"first_name\": string or null,\n"
                                "\"last_name\": string or null.\n\n"
                                
                                "Rules:\n"
                                "1. Use the value only if it is clearly shown in the document.\n"
                                "2. If a field is missing or unclear, keep its value as null.\n"
                                "3. Do NOT guess, infer, normalize, or reformat values.\n"
                                "4. Output ONLY the JSON object. No explanations, no extra text."
                            )
                        required_keys = ["first_name", "last_name"]
                        
                    elif folder_name == 'BirthCertificate':	
                        user_prompt_keywords_format2 = (
                                "Return ONE valid JSON object with exactly the following keys:\n"
                                "\"first_name\": string or null,\n"
                                "\"last_name\": string or null,\n"
                                "\"birthDate\": string or null (ONLY if clearly shown in format DD/MM/YYYY).\n\n"
                            
                                "Rules:\n"
                                "1. Use the value only if it is clearly shown in the document.\n"
                                "2. If a field is missing or unclear, keep its value as null.\n"
                                "3. Do NOT guess, infer, normalize, or reformat values.\n"
                                "4. Output ONLY the JSON object. No explanations, no extra text."
                            )
                        required_keys = ["first_name", "last_name","birthDate"]
                        
    
                    user_prompt_keywords2 =  "This is the document:" + out_image_Ocr + "\n" + user_prompt_keywords_format2
                    _, temp = OCR_extract(system_prompt_keywords2,user_prompt_keywords2,item,required_keys)
    
                    func = SAVE_FUNCS.get(folder_name)
                    func(csv_path=csv_path,js=json.loads(temp))
                    
                elif fname.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, fname)
                    print(f"Processing {pdf_path}")
            
                    medical_sources = {
                            "urls": [],
                            "pdf": [pdf_path]
                        }
                        
                    documents = build_medical_documents(medical_sources)
                    print("docs:", len(documents))
                    print(documents[0]["source"])
                    print(documents[0]["text"][:300])
                
                    # print("=== Text Embed ===")
                    # _, emb = TextEmbedTwin.analyze(
                    #     system_message=system_message_embedding,
                    #     user_message=documents[0]["text"]
                    # )
                      
                    # npy_path = pdf_path.replace(".pdf", ".npy")
                    # np.save(npy_path, emb)
                    # print(f"Saved embedding to {npy_path}")


                    # print("=== Text Keyword ===")
                    user_prompt_keywords_format2 = (
                                                    "Return ONE valid JSON object with exactly the following keys:\n"
                                                    "\"first_name\": string or null,\n"
                                                    "\"last_name\": string or null,\n"
                                                    "\"incomeAmount\": string or null.\n\n"
                                                
                                                    "Rules:\n"
                                                    "1. Use a value ONLY if it is explicitly and clearly shown in the document.\n"
                                                    "2. If a field is missing, unreadable, or uncertain, output null.\n"
                                                    "3. Do NOT guess, infer, normalize, convert, or reformat any value.\n"
                                                    "4. Output ONLY the JSON object. No explanations, no extra text."
                                                )
                    required_keys = ["first_name", "last_name","incomeAmount"]

    
                    user_prompt_keywords2 = "This is the document:" + documents[0]["text"] + "\n" + user_prompt_keywords_format2
                    item = []
                    _, temp = OCR_extract(system_prompt_keywords2,user_prompt_keywords2,item,required_keys)
                    json_to_Income(csv_path='/bask/homes/z/zhangyzz/KB/result/Income.csv', js=json.loads(temp))