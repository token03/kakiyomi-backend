import numpy as np
from typing import List, Dict, Optional
from .utils.textblock import TextBlock
from .rendering.render import cv2_to_pil
from .utils.translator_utils import encode_image_array, get_raw_text, set_texts_from_json, get_llm_client
from .utils.pipeline_utils import get_language_code
from deep_translator import GoogleTranslator, YandexTranslator, MicrosoftTranslator
import deepl

class Translator:
    def __init__(
        self,
        translator_tool_selection: str,
        source_lang: str = "",
        target_lang: str = "",
        credentials: Dict = None,
        lang_mapping: Dict = None,
        ui_translations: Dict = None, # Pass UI translations as dict
        llm_settings: Dict = None, # Pass LLM settings as dict
    ):
        """
        Initializes the Translator.

        Args:
            translator_tool_selection (str): Selected translation tool (localized name).
            source_lang (str, optional): Source language. Defaults to "".
            target_lang (str, optional): Target language. Defaults to "".
            credentials (Dict, optional): Dictionary of API credentials. Defaults to None.
            lang_mapping (Dict, optional): Dictionary for language code mapping. Defaults to None.
            ui_translations (Dict, optional): Dictionary of UI translations for tool names. Defaults to None.
            llm_settings (Dict, optional): Dictionary of LLM settings. Defaults to None.
        """
        self.credentials = credentials or {}
        self.lang_mapping = lang_mapping or {}
        self.ui_translations = ui_translations or {}
        self.llm_settings = llm_settings or {'image_input_enabled': False} # Default value if not provided


        self.translator_key = self.get_translator_key(translator_tool_selection)

        self.source_lang = source_lang
        self.source_lang_en = self.get_english_lang(self.source_lang)
        self.target_lang = target_lang
        self.target_lang_en = self.get_english_lang(self.target_lang)

        self.api_key = self.get_api_key(self.translator_key)
        self.api_url = self.get_api_url(self.translator_key)
        self.client = get_llm_client(self.translator_key, self.api_key, self.api_url)

        self.img_as_llm_input = self.llm_settings.get('image_input_enabled', False) # Use get with default in case it's not in dict


    def get_translator_key(self, localized_translator: str) -> str:
        """
        Maps localized translator names to keys.

        Args:
            localized_translator (str): Localized translator name.

        Returns:
            str: Translator key.
        """
        # Map localized translator names to keys
        translator_map = {
            self.ui_translations.get("Custom", "Custom"): "Custom", # Use get with default, fallback to "Custom" if translation missing
            self.ui_translations.get("Deepseek-v3", "Deepseek-v3"): "Deepseek-v3",
            self.ui_translations.get("GPT-4o", "GPT-4o"): "GPT-4o",
            self.ui_translations.get("GPT-4o mini", "GPT-4o mini"): "GPT-4o mini",
            self.ui_translations.get("Claude-3-Opus", "Claude-3-Opus"): "Claude-3-Opus",
            self.ui_translations.get("Claude-3.5-Sonnet", "Claude-3.5-Sonnet"): "Claude-3.5-Sonnet",
            self.ui_translations.get("Claude-3-Haiku", "Claude-3-Haiku"): "Claude-3-Haiku",
            self.ui_translations.get("Gemini-2.0-Flash", "Gemini-2.0-Flash"): "Gemini-2.0-Flash",
            self.ui_translations.get("Gemini-2.0-Pro", "Gemini-2.0-Pro"): "Gemini-2.0-Pro",
            self.ui_translations.get("Google Translate", "Google Translate"): "Google Translate",
            self.ui_translations.get("Microsoft Translator", "Microsoft Translator"): "Microsoft Translator",
            self.ui_translations.get("DeepL", "DeepL"): "DeepL",
            self.ui_translations.get("Yandex", "Yandex"): "Yandex"
        }
        return translator_map.get(localized_translator, localized_translator)

    def get_english_lang(self, translated_lang: str) -> str:
        """
        Gets the English language name from the translated language name.

        Args:
            translated_lang (str): Translated language name.

        Returns:
            str: English language name.
        """
        return self.lang_mapping.get(translated_lang, translated_lang)

    def get_llm_model(self, translator_key: str, ui_translations: Dict = None, credentials: Dict = None):
        """
        Gets the LLM model name based on the translator key.

        Args:
            translator_key (str): Translator key.
            ui_translations (Dict, optional): UI translations dict. Defaults to None.
            credentials (Dict, optional): Credentials dictionary. Defaults to None.

        Returns:
            str: LLM model name.
        """
        ui_translations = ui_translations or self.ui_translations
        credentials = credentials or self.credentials

        custom_model = credentials.get(ui_translations.get('Custom', 'Custom'), {}).get('model', '') # Use get with default

        model_map = {
            "Custom": custom_model,
            "Deepseek-v3": "deepseek-v3",
            "GPT-4o": "gpt-4o",
            "GPT-4o mini": "gpt-4o-mini",
            "Claude-3-Opus": "claude-3-opus-20240229",
            "Claude-3.5-Sonnet": "claude-3-5-sonnet-20241022",
            "Claude-3-Haiku": "claude-3-haiku-20240307",
            "Gemini-2.0-Flash": "gemini-2.0-flash",
            "Gemini-2.0-Pro": "gemini-2.0-pro-exp-02-05"
        }
        return model_map.get(translator_key)

    def get_system_prompt(self, source_lang: str, target_lang: str):
        """
        Gets the system prompt for LLM translation.

        Args:
            source_lang (str): Source language.
            target_lang (str): Target language.

        Returns:
            str: System prompt.
        """
        return f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.
        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.
        To aid you and provide context, You may be given the image of the page and/or extra context about the comic. You will be given a json string of the detected text blocks and the text to translate. Return the json string with the texts translated. DO NOT translate the keys of the json. For each block:
        - If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead
        - DO NOT give explanations
        Do Your Best! I'm really counting on you."""

    def get_deepseek_translation(self, user_prompt: str, system_prompt: str):
        """
        Gets translation using Deepseek API.

        Args:
            user_prompt (str): User prompt.
            system_prompt (str): System prompt.

        Returns:
            str: Translated text.
        """
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=message,
            temperature=0.7,
            max_tokens=1000,
        )

        translated = response.choices[0].message.content
        return translated

    def get_gpt_translation(self, user_prompt: str, model: str, system_prompt: str, image: np.ndarray):
        """
        Gets translation using GPT API.

        Args:
            user_prompt (str): User prompt.
            model (str): GPT model name.
            system_prompt (str): System prompt.
            image (np.ndarray): Input image.

        Returns:
            str: Translated text.
        """
        encoded_image = encode_image_array(image)

        if self.img_as_llm_input:
            message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}]}
            ]
        else:
            message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]

        response = self.client.chat.completions.create(
            model=model,
            messages=message,
            temperature=1,
            max_tokens=5000,
        )

        translated = response.choices[0].message.content
        return translated

    def get_claude_translation(self, user_prompt: str, model: str, system_prompt: str, image: np.ndarray):
        """
        Gets translation using Claude API.

        Args:
            user_prompt (str): User prompt.
            model (str): Claude model name.
            system_prompt (str): System prompt.
            image (np.ndarray): Input image.

        Returns:
            str: Translated text.
        """
        encoded_image = encode_image_array(image)
        media_type = "image/png"

        if self.img_as_llm_input:
            message = [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded_image}}]}
            ]
        else:
            message = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

        response = self.client.messages.create(
            model = model,
            system = system_prompt,
            messages=message,
            temperature=1,
            max_tokens=5000,
        )
        translated = response.content[0].text
        return translated

    def get_gemini_translation(self, user_prompt: str, model: str, system_prompt: str, image):
        """
        Gets translation using Gemini API.

        Args:
            user_prompt (str): User prompt.
            model (str): Gemini model name.
            system_prompt (str): System prompt.
            image (np.ndarray): Input image.

        Returns:
            str: Translated text.
        """

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 5000,
            }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
                },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
                },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
                },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
                },
        ]

        model_instance = self.client.GenerativeModel(model_name = model, generation_config=generation_config, system_instruction=system_prompt, safety_settings=safety_settings)
        chat = model_instance.start_chat(history=[])
        if self.img_as_llm_input:
            chat.send_message([image, user_prompt])
        else:
            chat.send_message([user_prompt])
        response = chat.last.text

        return response

    def translate(self, blk_list: List[TextBlock], image: np.ndarray, extra_context: str, ui_translations = None, credentials = None):
        """
        Translates text blocks using the selected translator.

        Args:
            blk_list (List[TextBlock]): List of text blocks to translate.
            image (np.ndarray): Input image.
            extra_context (str): Extra context for translation.
            ui_translations (Dict, optional): UI translations dict. Defaults to None.
            credentials (Dict, optional): Credentials dictionary. Defaults to None.

        Returns:
            List[TextBlock]: List of text blocks with translations.
        """
        ui_translations = ui_translations or self.ui_translations
        credentials = credentials or self.credentials

        source_lang_code = get_language_code(self.source_lang_en)
        target_lang_code = get_language_code(self.target_lang_en)

        # Non LLM Based
        if self.translator_key in ["Google Translate", "DeepL", "Yandex", "Microsoft Translator"]:
            for blk in blk_list:
                text = blk.text.replace(" ", "") if 'zh' in source_lang_code.lower() or source_lang_code.lower() == 'ja' else blk.text
                if self.translator_key == "Google Translate":
                    translation = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
                elif self.translator_key == "Yandex":
                    translation = YandexTranslator(source='auto', target=target_lang_code, api_key=self.api_key).translate(text)
                elif self.translator_key == "Microsoft Translator":
                    microsoft_credentials = credentials.get("Microsoft Azure", {}) # Get Microsoft credentials from passed credentials
                    region = microsoft_credentials.get('region_translator', '') # Get region
                    translation = MicrosoftTranslator(source_lang_code, target_lang_code, self.api_key, region).translate(text)
                elif self.translator_key == "DeepL":  # DeepL
                    trans = deepl.Translator(self.api_key)
                    target_lang_deepl = target_lang_code
                    if self.target_lang == ui_translations.get("Simplified Chinese", "Simplified Chinese"): # Use get with default
                        target_lang_deepl = "zh"
                    elif self.target_lang == ui_translations.get("English", "English"): # Use get with default
                        target_lang_deepl = "EN-US"
                    result = trans.translate_text(text, source_lang=source_lang_code, target_lang=target_lang_deepl)
                    translation = result.text

                if translation is not None:
                    blk.translation = translation

        # Handle LLM based translations
        else:
            model = self.get_llm_model(self.translator_key, ui_translations, credentials)
            entire_raw_text = get_raw_text(blk_list)
            system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
            user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"

            if 'Custom' in self.translator_key:
                entire_translated_text = self.get_gpt_translation(user_prompt, model, system_prompt, image)
            elif 'Deepseek' in self.translator_key:
                entire_translated_text = self.get_deepseek_translation(user_prompt, system_prompt)
            elif 'GPT' in self.translator_key:
                entire_translated_text = self.get_gpt_translation(user_prompt, model, system_prompt, image)
            elif 'Claude' in self.translator_key:
                entire_translated_text = self.get_claude_translation(user_prompt, model, system_prompt, image)
            elif 'Gemini' in self.translator_key:
                image_pil = cv2_to_pil(image) # Convert image here before passing
                entire_translated_text = self.get_gemini_translation(user_prompt, model, system_prompt, image_pil)

            set_texts_from_json(blk_list, entire_translated_text)

        return blk_list

    def get_api_key(self, translator_key: str, ui_translations = None, credentials = None):
        """
        Gets the API key for the specified translator.

        Args:
            translator_key (str): Translator key.
            ui_translations (Dict, optional): UI translations dict. Defaults to None.
            credentials (Dict, optional): Credentials dictionary. Defaults to None.

        Returns:
            str: API key.
        """
        ui_translations = ui_translations or self.ui_translations
        credentials = credentials or self.credentials

        api_key = ""

        if 'Custom' in translator_key:
            api_key = credentials.get(ui_translations.get('Custom', 'Custom'), {}).get('api_key', "") # Use get with default
        elif 'Deepseek' in translator_key:
            api_key = credentials.get(ui_translations.get('Deepseek', 'Deepseek'), {}).get('api_key', "") # Use get with default
        elif 'GPT' in translator_key:
            api_key = credentials.get(ui_translations.get('Open AI GPT', 'Open AI GPT'), {}).get('api_key', "") # Use get with default
        elif 'Claude' in translator_key:
            api_key = credentials.get(ui_translations.get('Anthropic Claude', 'Anthropic Claude'), {}).get('api_key', "") # Use get with default
        elif 'Gemini' in translator_key:
            api_key = credentials.get(ui_translations.get('Google Gemini', 'Google Gemini'), {}).get('api_key', "") # Use get with default
        else:
            api_key_map = {
                "Microsoft Translator": credentials.get("Microsoft Azure", {}).get('api_key_translator', ""), # Get from passed credentials
                "DeepL": credentials.get(ui_translations.get('DeepL', 'DeepL'), {}).get('api_key', ""), # Use get with default
                "Yandex": credentials.get(ui_translations.get('Yandex', 'Yandex'), {}).get('api_key', ""), # Use get with default
            }
            api_key = api_key_map.get(translator_key, "")

        if translator_key == 'Google Translate' or translator_key == 'Custom':
            pass
        elif not api_key:
            raise ValueError(f"API key not found for translator: {translator_key}")

        return api_key

    def get_api_url(self, translator_key: str, ui_translations = None, credentials = None):
        """
        Gets the API URL for the specified translator.

        Args:
            translator_key (str): Translator key.
            ui_translations (Dict, optional): UI translations dict. Defaults to None.
            credentials (Dict, optional): Credentials dictionary. Defaults to None.

        Returns:
            str: API URL.
        """
        ui_translations = ui_translations or self.ui_translations
        credentials = credentials or self.credentials
        api_url = ""

        if 'Custom' in translator_key:
            api_url = credentials.get(ui_translations.get('Custom', 'Custom'), {}).get('api_url', "") # Use get with default

        return api_url