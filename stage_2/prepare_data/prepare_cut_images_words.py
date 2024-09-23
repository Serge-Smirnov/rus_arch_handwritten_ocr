from PIL import Image, ImageDraw
from typing import List
import json
from json import JSONEncoder
from tqdm import tqdm
import os


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Bbox():
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class Word():
    def __init__(self, value: str, bbox: Bbox, segmentation: List[float]) -> None:
        self.value=value
        self.bbox = bbox
        self.segmentation = segmentation


class CocoJson():
    def __init__(self, path: str, file_name: str, width: float, height: float, words: List[Word]) -> None:
        self.path = path
        self.file_name = file_name
        self.original_width = width
        self.original_height = height
        self.words = words


def get_poligon(raw_poligon:[]) -> []:
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


# @profile
def cut_image_part(image_name: str, poligon: [], bbox: Bbox, saved_name: str):
    try:
        with Image.open(image_name) as im:
            poligon = get_poligon(poligon)

            mask = Image.new("L", im.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(poligon, fill=255, outline=None)
            black = Image.new("L", im.size, 0)
            result = Image.composite(im, black, mask)

            bbox = (bbox.x, bbox.y, bbox.x+bbox.width, bbox.y+bbox.height)
            new_img = result.crop(bbox)
            new_img.save(saved_name)

    except Exception as e:
        print(e.__traceback__, e)


# @profile
def parseCocoJson(full_file_name: str) -> []:
    result = []

    with open(full_file_name) as json_file:
        data = json.load(json_file)

        annotations = data['annotations']

        for image in tqdm(data['images']):
            image_id = image['id']

            words = []
            # nested loop filter
            for item in annotations:
                metadata = item["metadata"]
                if "name" in metadata and item['image_id'] == image_id:
                    name = metadata["name"]
                    original_bbox = item['bbox']
                    words.append(Word(value=name,
                                      bbox=Bbox(x=original_bbox[0], 
                                                y=original_bbox[1], 
                                                width=original_bbox[2], 
                                                height=original_bbox[3]),
                                      segmentation=item['segmentation'][0]))
            if len(words) > 0:
                cocoJson = CocoJson(path=image['path'],
                                    file_name=image['file_name'],
                                    width=image['width'],
                                    height=image['height'],
                                    words=words)
                result.append(cocoJson)

    return result


# @profile
def prepare_images_for_dataset(json_full_file_name: str, raw_images: str, dataset_dir_name: str):

    metadata = parseCocoJson(json_full_file_name)

    # with open('./stage_2/prepare_data/stage_2_annotations.json', 'w') as file:
    #     json.dump({"annotations": metadata}, file,
    #               indent=2, cls=CustomEncoder)
    labels_stage_2 = {}
    for image in tqdm(metadata):
        for idx, word in enumerate(image.words):

            raw_image_name = f'{raw_images}/{image.file_name}'
            image_name = dataset_dir_name + f'/{image.file_name}_{idx}.jpg'     

            # форомируем файл аннотаций/леблов для второго этапа - перевод. Где картинка-слово сопоставляется переводу котоый написал специалист по разметке данных
            labels_stage_2.update({image_name: word.value})

            # сохраням вырезанное слово как ратинку
            if not os.path.exists(image_name):
                directory = os.path.dirname(image_name)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                cut_image_part(image_name=raw_image_name,
                               poligon=word.segmentation,
                               bbox=word.bbox,
                               saved_name=image_name)

    with open(dataset_dir_name + f'/label.json', 'w') as file:
        json.dump(labels_stage_2, file, indent=2, cls=CustomEncoder)


if __name__ == '__main__':
    prepare_images_for_dataset(
        json_full_file_name='./hcr-archive-metric-book/data_mb/coco/merged.json',
        raw_images='./hcr-archive-metric-book/data_mb/images',
        dataset_dir_name="./hcr-archive-metric-book/data_mb/word_images")
