import io
import json
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
from modules.utils.textblock import TextBlock
from services.detection_service import DetectionService
from services.inpainting_service import InpaintingService
from services.ocr_service import OCRService
from services.translation_service import TranslationService
from services.utils.pipeline_utils import uploadfile_to_nparray

class SegmentResponse(BaseModel):
    bounding_boxes: List[List[int]]

class OCRResponse(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    translation_list: List[str]

class InpaintResponse(BaseModel):
    image: str

translation_service = None
ocr_service = None
detection_service = None
inpainting_service = None

async def lifespan(app: FastAPI):
    global translation_service, ocr_service, detection_service, inpainting_service
    translation_service = TranslationService()
    ocr_service = OCRService()
    detection_service = DetectionService()
    inpainting_service = InpaintingService()
     
    yield
    detection_service.clean_up()

app = FastAPI(lifespan=lifespan)

@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    source_language: str = Form("Japanese"),
):
    print("Segmenting image")

    np_image = await uploadfile_to_nparray(image)

    blk_list = detection_service.detect(np_image, source_language)

    formatted_results = []
    for blk in blk_list:
        formatted_result = {
            "xyxy": blk.xyxy.tolist(),
            "bubble_xyxy": blk.bubble_xyxy.tolist() if blk.bubble_xyxy is not None else [],
            "inpaint_bboxes": [[inpbb[0].item(), inpbb[1].item(), inpbb[2].item(), inpbb[3].item()] for inpbb in blk.inpaint_bboxes],
            "txt_class": blk.text_class
        }
        formatted_results.append(formatted_result)

    return {"segment_list": formatted_results}


@app.post("/ocr")
async def ocr(
    source_language: str = Form("Japanese"),
    image: UploadFile = File(...), 
    segment_json: str = Form(None)
):
    print("Performing OCR")

    segments = json.loads(segment_json)["segment_list"]

    blk_list = []
    for seg in segments:
        txt_bbox = np.array(seg["xyxy"], dtype=np.int32) if seg["xyxy"] else None
        bble_bbox = np.array(seg["bubble_xyxy"], dtype=np.int32) if seg["bubble_xyxy"] else None
        
        inp_bboxes = [tuple(np.int32(b[i]) for i in range(4)) for b in seg["inpaint_bboxes"]]

        blk_list.append(TextBlock(
            txt_bbox, 
            bble_bbox, 
            seg["txt_class"], 
            inp_bboxes
        ))

    np_image = await uploadfile_to_nparray(image)

    ocr_results = ocr_service.process(np_image, blk_list, source_language)

    formatted_results = [result.text for result in ocr_results]

    return {"processed_text": formatted_results}

@app.post("/translate")
async def translate(
    # filler values, will be replaced with actual values
    blk_json: str = Form('{"blk_list": ["こんにちは", "元気ですか？", "元気です", "ありがとう"]}'),
    source_language: str = Form("Japanese"),
    target_language: str = Form("English"),
    image_context: UploadFile = File(None),
    extra_context: Optional[str] = Form(""),
    translator: Optional[str] = Form("Custom"),
    custom_model: Optional[str] = Form("google/gemini-2.0-flash-lite-preview-02-05:free"),
    custom_url: Optional[str] = Form("https://openrouter.ai/api/v1")
):
    print("Translating text")

    image = None
    if image_context:
        image = await uploadfile_to_nparray(image_context)

    blk_list = json.loads(blk_json)["blk_list"]

    translation_object = translation_service.translate_text(
        blk_list=blk_list,
        source_lang=source_language,
        target_lang=target_language,
        image=image,
        extra_context=extra_context,
        translator=translator,
        custom_model=custom_model,
        custom_url=custom_url,
    )

    formatted_results = [translation_object[key] for key in translation_object]

    return {"translation_list": formatted_results}

@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...), 
    mask: str = Form(...), 
):
    print("Inpainting image")

    return {"message": "Inpainting successful"}