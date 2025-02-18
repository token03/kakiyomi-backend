import cv2
import base64
import json
import re
from fastapi import HTTPException
import stanza
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import anthropic
from .textblock import TextBlock
from typing import List


import cv2
import base64
import json
import re
import stanza
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import anthropic
# from .textblock import TextBlock # Removed TextBlock import
from typing import List


def encode_image_array(img_array: np.ndarray):
    _, img_bytes = cv2.imencode('.png', img_array)
    return base64.b64encode(img_bytes).decode('utf-8')

def get_llm_client(translator: str, api_key: str, api_url: str = ""):
    if 'Custom' in translator:
        client = OpenAI(api_key=api_key, base_url=api_url)
    elif 'Deepseek' in translator:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    elif 'GPT' in translator:
        client  = OpenAI(api_key = api_key)
    elif 'Claude' in translator:
        client = anthropic.Anthropic(api_key = api_key)
    elif 'Gemini' in translator:
        client = genai
        client.configure(api_key = api_key)
    else:
        client = None

    return client

def get_raw_translation(blk_list: List[str]):
    raw_translations = "\n".join(blk for blk in blk_list)
    return raw_translations

def process_translation_response(response_string):
    try:
        translation_list = json.loads(response_string)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {response_string}")
        raise HTTPException(status_code=500, detail=f"Invalid response from translation service: {str(e)}")
    return translation_list

def format_translation_request(blk_list: List[str]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        key = f"block_{idx}"
        rw_txts_dict[key] = {"text": blk}
    
    return json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)

def set_upper_case(blk_list: List[str], upper_case: bool): # Changed TextBlock to str
    for idx, blk in enumerate(blk_list):
        if upper_case and not blk.isupper():
            blk_list[idx] = blk.upper()
        elif not upper_case and blk.isupper():
            blk_list[idx] = blk.capitalize()
        else:
            blk_list[idx] = blk

def format_translations(blk_list: List[str], trg_lng_cd: str, upper_case: bool =True): # Changed TextBlock to str
    for idx, blk in enumerate(blk_list):
        if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):

            if trg_lng_cd == 'zh-TW':
                trg_lng_cd = 'zh-Hant'
            elif trg_lng_cd == 'zh-CN':
                trg_lng_cd = 'zh-Hans'
            else:
                trg_lng_cd = trg_lng_cd

            stanza.download(trg_lng_cd, processors='tokenize')
            nlp = stanza.Pipeline(trg_lng_cd, processors='tokenize')
            doc = nlp(blk)
            seg_result = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    seg_result.append(word.text)
            translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
            blk_list[idx] = translation
        else:
            set_upper_case(blk_list, upper_case)

def is_there_text(blk_list: List[str]) -> bool: # Changed TextBlock to str
    return any(blk for blk in blk_list)