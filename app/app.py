import json
import traceback
from distutils.util import strtobool

import cv2
import numpy as np
import uvicorn
from PIL import Image, ImageDraw
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi import File
from starlette_prometheus import PrometheusMiddleware, metrics
from typing_extensions import Annotated

from load_model_wigth import load_segmentation_weights, load_translation_weights
from segmentation_model.model import get_segmentor_instance, Segmentor
from translation_model.model import Translator, get_translator_instance

config = dotenv_values(".env")

app = FastAPI(
    title="Движок распознавания рукописного текста",
    description="""Сервис для распознавания рукописного текста метрических книг."""
)

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

segmentor: Segmentor = None
translator: Translator = None


def __init_models__():
    global segmentor
    global translator

    if not segmentor:
        model_path = f'{config["segmentation_dir"]}/model_weight.pth'
        config_path = f'{config["segmentation_dir"]}/config.yaml'
        segmentor = get_segmentor_instance(model_path=model_path, config_path=config_path)

    if not translator:
        with open(f'{config["translation_dir"]}/config.json', encoding='utf-8') as json_file:
            translator = get_translator_instance(
                model_path=f'{config["translation_dir"]}/model_weight.ckpt',
                resnet_model_path=f'{config["translation_dir"]}/resnet34.pth',
                config=json.load(json_file))


def __load_models_and_weights__():
    #  загрузка весов модели
    load_segmentation_weights(config["segmentation_model_weight_path"], config["segmentation_dir"])
    load_translation_weights(model_weight_url=config["translation_model_weight_path"],
                             resnet_weight_url=config["resnet_model_weight_path"],
                             directory=config["translation_dir"])


def __init__():
    if strtobool(config["load_model_wights_from_bin_repo"]):
        __load_models_and_weights__()

    __init_models__()


def recognize(image) -> dict:
    coco_data = segmentor.segment(image)
    # картинка тут одна
    for annotation in coco_data['annotations']:
        original_bbox = annotation['bbox']
        segmentations = annotation['segmentation']

        im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = Image.new("L", im.size, 0)
        draw = ImageDraw.Draw(mask)
        has_poligon = False
        for poligon in segmentations:
            poligon = get_poligon(poligon)
            if len(poligon) > 2:
                draw.polygon(poligon, fill=255, outline=None)
                has_poligon = True
        if not has_poligon:
            continue

        black = Image.new("L", im.size, 0)
        result = Image.composite(im, black, mask)

        bbox = (original_bbox[0],
                original_bbox[1],
                original_bbox[0] + original_bbox[2],
                original_bbox[1] + original_bbox[3])
        new_img = result.crop(bbox)
        image_cv2 = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
        predicted_word = translator.translate(image_cv2)
        metadata = annotation["metadata"]
        del annotation["segmentation"]  # а можно и не удалять
        metadata['name'] = predicted_word
    coco_data['hocr'] = to_hocr(coco_data)
    return coco_data


def to_hocr(coco_data: dict) -> str:
    image_info = coco_data['images'][0]
    hocr_content = """<!DOCTYPE html><html><head><title>{image_info['file_name']}</title></head><body>"""

    hocr_content += f"<div class='ocr_page' id='page_{image_info['id']}' title='bbox 0 0 {image_info['width']} {image_info['height']}'>"
    hocr_content += f"<div class='ocr_carea' id='block_{image_info['id']}'>"
    hocr_content += f"<p class='ocr_par' id='par_{image_info['id']}' lang='rus' title='bbox 0 0 {image_info['width']} {image_info['height']}'>"

    for annotation in coco_data['annotations']:
        # Assuming 'bbox' contains [x, y, width, height]
        bbox = annotation['bbox']

        # Extracted text
        text = annotation['metadata']['name']
        hocr_content += f"<span class='ocrx_word' id='word_{image_info['id']}_{annotation['id']}' " \
                        f"title='bbox {bbox[0]} {bbox[1]} {bbox[0] + bbox[2]} {bbox[1] + bbox[3]}'>{text}</span>"

    hocr_content += """</p></div></div></body></html>"""

    return hocr_content


def get_poligon(raw_poligon: []) -> []:
    result = []
    tuples = []
    for i, item in enumerate(raw_poligon):
        if i % 2 == 0 and i != 0:
            result.append((tuples[0], tuples[1]))
            tuples = []
            tuples.append(item)
        else:
            tuples.append(item)

    return result


@app.post('/segmentor', summary="Сегментация образа")
async def translate_image_word(file: Annotated[bytes, File()]):
    if len(file) == 0:
        raise HTTPException(status_code=400, detail="Картинка имеет нулевой размер байтов")
    try:
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        return segmentor.segment(img)
    except Exception:
        return None


@app.post('/translate_word_image', summary="Перевод кусочка изображения со словом (результат работы этапа сегментации)")
async def translate_image_word(file: Annotated[bytes, File()]):
    if len(file) == 0:
        raise HTTPException(status_code=400, detail="Картинка имеет нулевой размер байтов")
    try:
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        return translator.translate(img)
    except Exception:
        return None


@app.post('/recognize', summary="Распознать образ")
async def translate_image_word(file: Annotated[bytes, File()]):
    if len(file) == 0:
        raise HTTPException(status_code=400, detail="Картинка имеет нулевой размер байтов")
    try:
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        return recognize(image=img)
    except Exception as e:
        traceback.print_exc()
        return None


if __name__ == '__main__':
    __init__()
    uvicorn.run(app, host='0.0.0.0', port=8087)
