import json
import os
from PIL import Image
import cv2
from dotenv import load_dotenv
import numpy as np
from typing import List, Optional
from .utils.translator_utils import encode_image_array, format_translation_request, get_llm_client, process_translation_response
from .utils.pipeline_utils import get_language_code
from deep_translator import GoogleTranslator

load_dotenv()

class TranslationService:
    def __init__(self):
        print("In")
        self.CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
        self.CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.GPT_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        self.MODEL_CONFIG = {
            "Deepseek-v3": "deepseek-v3",
            "GPT-4o": "gpt-4o",
            "GPT-4o mini": "gpt-4o-mini",
            "Claude-3-Opus": "claude-3-opus-20240229",
            "Claude-3.5-Sonnet": "claude-3-5-sonnet-20241022",
            "Claude-3-Haiku": "claude-3-haiku-20240307",
            "Gemini-2.0-Flash": "gemini-2.0-flash",
            "Gemini-2.0-Pro": "gemini-2.0-pro-exp-02-05"
        }

        self.CLIENT_CONFIG = {
            "Custom": "Custom",
            "Deepseek-v3": "Deepseek",
            "GPT-4o": "GPT",
            "GPT-4o mini": "GPT",
            "Claude-3-Opus": "Claude",
            "Claude-3.5-Sonnet": "Claude",
            "Claude-3-Haiku": "Claude",
            "Gemini-2.0-Flash": "Gemini",
            "Gemini-2.0-Pro": "Gemini" 
        }
        print("Translation Service initialized")

    def get_key(self, translator: str) -> str:
        if translator == "Custom":
            return self.CUSTOM_API_KEY
        elif translator == "Deepseek":
            return self.DEEPSEEK_API_KEY
        elif translator == "GPT":
            return self.GPT_API_KEY
        elif translator == "Claude":
            return self.CLAUDE_API_KEY
        elif translator == "Gemini":
            return self.GOOGLE_API_KEY
        else:
            return ""

    def get_model(self, translator: str) -> str:
        return self.MODEL_CONFIG.get(translator, "")

    def get_client_name(self, translator: str) -> str:
        return self.CLIENT_CONFIG.get(translator, "")

    def get_system_prompt(self, source_lang: str, target_lang: str) -> str:
        return f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.
        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.
        To aid you and provide context, You may be given the image of the page and/or extra context about the comic.
        You will be given a json string of the detected text blocks 
        Return an OBJECT with corresponding index keys and translated text values. (block_X: "translated text")
        - If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead
        - DO NOT give explanations
        Do Your Best! I'm really counting on you."""
    def get_google_translation(self, text: str, source_lang_code: str, target_lang_code: str) -> str:
        if 'zh' in source_lang_code.lower() or source_lang_code.lower() == 'ja':
            text = text.replace(" ", "")
        return GoogleTranslator(source='auto', target=target_lang_code).translate(text)

    def get_llm_translation(self, user_prompt: str, system_prompt: str, image: Optional[np.ndarray], translator: str, custom_model: str, custom_url: str) -> str:
        client_name = self.get_client_name(translator)
        if not client_name:
            raise ValueError(f"No LLM configuration found for translator '{translator}'.")

        api_key = self.get_key(client_name)
        client = get_llm_client(translator=translator, api_key=api_key, api_url=custom_url)

        encoded_image = None
        if image is not None:
            encoded_image = encode_image_array(image)
            img_as_llm_input = True

        img_as_llm_input = False
        if img_as_llm_input:
            message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]}
            ]
        else:
            message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]

        model = self.get_model(translator)
        if not model:
            model = custom_model

        response = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=1,
            max_tokens=5000,
            response_format={
                "type": "json_object"
            }
        )
        translated = response.choices[0].message.content
        return translated

    def translate_text(
        self,
        blk_list: List[str],
        image: np.ndarray,
        extra_context: str,
        translator: str,
        source_lang: str,
        target_lang: str,
        custom_url: Optional[str] = None,
        custom_model: Optional[str] = None,
    ) -> List[str]:
        source_lang_code = get_language_code(source_lang)
        target_lang_code = get_language_code(target_lang)

        if translator == "google_translate":
            for blk in blk_list:
                translated_text = self.get_google_translation(blk.text, source_lang_code, target_lang_code)
                blk.translation = translated_text
        elif translator in self.CLIENT_CONFIG:
            entire_raw_text = format_translation_request(blk_list)
            print(entire_raw_text)
            system_prompt = self.get_system_prompt(source_lang, target_lang)
            user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"
            translation_list = self.get_llm_translation(user_prompt, system_prompt, image, translator, custom_model, custom_url)
        else:
            raise NotImplementedError(f"Translator '{translator}' is not supported.")

        return process_translation_response(translation_list)