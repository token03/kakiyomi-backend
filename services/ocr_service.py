import os
import cv2
import json
import base64
from dotenv import load_dotenv
import requests
import numpy as np
import easyocr

from paddleocr import PaddleOCR
from typing import List

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from .utils.textblock import TextBlock, adjust_text_line_coordinates, sort_blk_list
from .utils.pipeline_utils import lists_to_blk_list, language_codes
from .utils.download import get_models, manga_ocr_data, pororo_data
from .ocr.manga_ocr.manga_ocr import MangaOcr
from .ocr.pororo.main import PororoOcr

# Temporary configuration (to be moved to a YAML config later)
DEFAULT_SOURCE_LANG = "Japanese"          # e.g., "English", "Japanese", "Chinese", "Korean"
DEFAULT_OCR_MODEL = "Default"              # Options: "Default", "Microsoft OCR", "Google Cloud Vision"
DEFAULT_DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

load_dotenv()

DEFAULT_CREDENTIALS = {
    "api_key_ocr": "YOUR_MICROSOFT_OCR_API_KEY",
    "endpoint": "YOUR_MICROSOFT_OCR_ENDPOINT",
    "api_key": "YOUR_GOOGLE_API_KEY"
}

class OCRService:
    def __init__(self,
                 source_lang: str = DEFAULT_SOURCE_LANG,
                 ocr_model: str = DEFAULT_OCR_MODEL,
                 device: str = DEFAULT_DEVICE,
                 credentials: dict = DEFAULT_CREDENTIALS,
                 gpt_client=None):
        print("Initializing OCR Service")
        self.source_lang = source_lang
        self.ocr_model = ocr_model
        self.device = device
        self.credentials = credentials
        self.gpt_client = gpt_client  # Should be a valid client instance if using GPT OCR

        # Determine OCR flags based on configuration
        self.microsoft_ocr = (self.ocr_model == "Microsoft OCR")
        self.google_ocr = (self.ocr_model == "Google Cloud Vision")
        self.use_paddle_for_chinese = (self.source_lang == "Chinese" and not (self.microsoft_ocr or self.google_ocr))
        # For some European languages, if using the default OCR, we opt to use GPT OCR.
        if self.source_lang in ["French", "German", "Dutch", "Russian", "Spanish", "Italian"] and self.ocr_model == "Default":
            self.gpt_ocr = True
        else:
            self.gpt_ocr = False

        # Pre-load OCR models as needed
        if self.use_paddle_for_chinese:
            self.paddle_ocr = PaddleOCR(lang='ch')
        else:
            self.paddle_ocr = None

        if self.source_lang == "Japanese" and self.ocr_model == "Default":
            get_models(manga_ocr_data)
            manga_ocr_path = './models/ocr/manga-ocr-base'
            self.manga_ocr = MangaOcr(pretrained_model_name_or_path=manga_ocr_path, device=self.device)
        else:
            self.manga_ocr = None

        if self.source_lang == "English" and self.ocr_model == "Default":
            self.easyocr_reader = easyocr.Reader(['en'], gpu=(self.device != 'cpu'))
        else:
            self.easyocr_reader = None

        if self.source_lang == "Korean" and self.ocr_model == "Default":
            get_models(pororo_data)
            self.pororo_ocr = PororoOcr()
        else:
            self.pororo_ocr = None

        print("OCR Service initialized")

    def set_source_orientation(self, blk_list: List[TextBlock]):
        # Sets a language code for each text block based on the language_codes mapping.
        source_lang_code = language_codes.get(self.source_lang, self.source_lang)
        for blk in blk_list:
            blk.source_lang = source_lang_code

    def process(self, img: np.ndarray, blk_list: List[TextBlock], source_lang: str = None) -> List[TextBlock]:
        if source_lang is not None:
            self.source_lang = source_lang
            # Optionally update related flags if needed
            self.use_paddle_for_chinese = (self.source_lang == "Chinese" and not (self.microsoft_ocr or self.google_ocr))
            self.gpt_ocr = (self.source_lang in ["French", "German", "Dutch", "Russian", "Spanish", "Italian"] 
                            and self.ocr_model == "Default")

        print(f"Processing Image with source language: {self.source_lang}")

        self.set_source_orientation(blk_list)

        # Choose the OCR method based on current configuration
        if self.use_paddle_for_chinese:
            return self._ocr_paddle(img, blk_list)
        elif self.microsoft_ocr:
            return self._ocr_microsoft(img, blk_list)
        elif self.google_ocr:
            return self._ocr_google(img, blk_list)
        elif self.gpt_ocr:
            return self._ocr_gpt(img, blk_list)
        else:
            return self._ocr_default(img, blk_list)

    def _ocr_paddle(self, img: np.ndarray, blk_list: List[TextBlock]) -> List[TextBlock]:
        result = self.paddle_ocr.ocr(img) if self.paddle_ocr else []
        if result:
            result = result[0]  # PaddleOCR returns a nested list
        texts_bboxes = [tuple(coord for point in bbox for coord in point) for bbox, _ in result] if result else []
        # Condense bounding boxes to (x1, y1, x2, y2)
        condensed_texts_bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y1_, x2_, y2, x1_, y2_) in texts_bboxes] if texts_bboxes else []
        texts_string = [line[1][0] for line in result] if result else []
        blk_list = lists_to_blk_list(blk_list, condensed_texts_bboxes, texts_string)
        return blk_list

    def _ocr_microsoft(self, img: np.ndarray, blk_list: List[TextBlock]) -> List[TextBlock]:
        texts_bboxes = []
        texts_string = []

        api_key = self.credentials.get('api_key_ocr')
        endpoint = self.credentials.get('endpoint')
        if not api_key or not endpoint:
            print("Microsoft OCR credentials not provided.")
            return blk_list

        client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        image_buffer = cv2.imencode('.png', img)[1].tobytes()
        result = client.analyze(image_data=image_buffer, visual_features=[VisualFeatures.READ])

        if result.read is not None and result.read.blocks:
            # Using the first block as in the original logic
            for line in result.read.blocks[0].lines:
                vertices = line.bounding_polygon
                if all('x' in vertex and 'y' in vertex for vertex in vertices):
                    x1 = vertices[0]['x']
                    y1 = vertices[0]['y']
                    x2 = vertices[2]['x']
                    y2 = vertices[2]['y']
                    texts_bboxes.append((x1, y1, x2, y2))
                    texts_string.append(line.text)
        blk_list = lists_to_blk_list(blk_list, texts_bboxes, texts_string)
        return blk_list

    def _ocr_google(self, img: np.ndarray, blk_list: List[TextBlock]) -> List[TextBlock]:
        texts_bboxes = []
        texts_string = []

        api_key = self.credentials.get('api_key')
        if not api_key:
            print("Google OCR API key not provided.")
            return blk_list

        ret, buffer = cv2.imencode('.png', img)
        image_content = base64.b64encode(buffer).decode('utf-8')
        payload = {
            "requests": [{
                "image": {"content": image_content},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            "https://vision.googleapis.com/v1/images:annotate",
            headers=headers,
            params={"key": api_key},
            data=json.dumps(payload)
        )
        result = response.json()
        texts = result.get('responses', [{}])[0].get('textAnnotations', [])
        if texts:
            for index, text in enumerate(texts):
                vertices = text.get('boundingPoly', {}).get('vertices', [])
                if index == 0:
                    continue  # skip the full text annotation
                if all('x' in vertex and 'y' in vertex for vertex in vertices):
                    x1 = vertices[0]['x']
                    y1 = vertices[0]['y']
                    x2 = vertices[2]['x']
                    y2 = vertices[2]['y']
                    texts_bboxes.append((x1, y1, x2, y2))
                    texts_string.append(text.get('description', ''))
        blk_list = lists_to_blk_list(blk_list, texts_bboxes, texts_string)
        return blk_list

    def _ocr_gpt(self, img: np.ndarray, blk_list: List[TextBlock], expansion_percentage: int = 0) -> List[TextBlock]:
        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(blk.xyxy, expansion_percentage, expansion_percentage, img)
            # Validate coordinates
            if x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                ret, buf = cv2.imencode('.png', cropped_img)
                base64_image = base64.b64encode(buf).decode('utf-8')
                text = get_gpt_ocr(base64_image, self.gpt_client) if self.gpt_client else ""
                blk.text = text
        return blk_list

    def _ocr_default(self, img: np.ndarray, blk_list: List[TextBlock], expansion_percentage: int = 5) -> List[TextBlock]:
        gpu_state = False if self.device == 'cpu' else True
        for blk in blk_list:
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(blk.xyxy, expansion_percentage, expansion_percentage, img)
            if x1 < x2 and y1 < y2:
                if self.source_lang == "Japanese":
                    if self.manga_ocr is not None:
                        blk.text = self.manga_ocr(img[y1:y2, x1:x2])
                elif self.source_lang == "English":
                    if self.easyocr_reader is not None:
                        result = self.easyocr_reader.readtext(img[y1:y2, x1:x2], paragraph=True)
                        texts = [r[1] for r in result if r is not None]
                        blk.text = ' '.join(texts)
                elif self.source_lang == "Korean":
                    if self.pororo_ocr is not None:
                        self.pororo_ocr.run_ocr(img[y1:y2, x1:x2])
                        result = self.pororo_ocr.get_ocr_result()
                        descriptions = result.get('description', [])
                        blk.text = ' '.join(descriptions)
            else:
                print('Invalid text bbox for target image')
                blk.text = ''
        return blk_list

def get_gpt_ocr(base64_image: str, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "Write out the text in this image. Do NOT Translate. Do not write anything else"}
                ]
            }
        ],
        max_tokens=1000,
    )
    text = response.choices[0].message.content
    text = text.replace('\n', ' ') if '\n' in text else text
    return text
