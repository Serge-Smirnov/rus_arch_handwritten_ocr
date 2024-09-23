import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil


def get_coco_files(dir_with_coco_files: str):
    coco_files = []
    for file_name in os.listdir(dir_with_coco_files):
        if os.path.isfile(os.path.join(dir_with_coco_files, file_name)) and file_name.lower().endswith(".json"):
            coco_files.append(os.path.join(dir_with_coco_files, file_name))
    print(coco_files)
    return coco_files

def save_to_file(coco_obj: dict, output_file: str):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_file, "w") as f:
        json.dump(coco_obj, f)


def merge(coco_files: list):
    output = {"images": [], "categories": [], "annotations": []}
    image_ids = []
    ann_ids = []

    for coco_file in tqdm(coco_files):
        with open(coco_file, "r") as f:
            coco_data = json.load(f)

        # категория у нас всего одна, потому не замарачиваемся с мержем категорий
        if len(output["categories"]) == 0:
            output["categories"].extend(coco_data["categories"])

        # мержим картинки
        for image in coco_data["images"]:
            image_id = image["id"]
            # если номер картинки пересекается, то задаем новый номер и меняем номер в аннотациях
            if image_id in image_ids:
                new_image_id = max(image_ids) + 1
                image["id"] = new_image_id
                for ann in coco_data["annotations"]:
                    if ann["image_id"] == image_id:
                        ann["image_id"] = new_image_id

            output["images"].append(image)
            image_ids.append(image["id"])

        # мержим аннотации
        for ann in coco_data["annotations"]:
            ann_id = ann["id"]
            # если номер аннотации пересекается, то задаем новый номер
            if ann_id in ann_ids:
                new_ann_id = max(ann_ids) + 1
                ann["id"] = new_ann_id

            output["annotations"].append(ann)
            ann_ids.append(ann["id"])

    return output


def find_relative_path(base_dir: str, sub_path: str):
    if os.path.exists(os.path.join(base_dir, sub_path)):
        return sub_path

    split_path = sub_path.split("/", 1)
    while len(split_path) > 1:
        if os.path.exists(os.path.join(base_dir, split_path[1])):
            return split_path[1]
        split_path = split_path[1].split("/", 1)

    return None


def fix_image_paths(coco_obj: dict, image_dir: str):
    images = coco_obj["images"]
    for image in images:
        path = find_relative_path(image_dir, image["path"])
        if path is not None:
            image["path"] = path
            image["file_name"] = path
            
            

def prepare_train_and_val_coco_files(source_coco_filename: str):
    # Подгрузим аннотации
    with open(source_coco_filename) as f:
        annotations = json.load(f)

    # Пустые словари для аннотаций обучения и валидации
    annotations_train, annotations_val = {}, {}

    # скопируем категории в словарь обучения и валидации (у нас всего одна категория 'word')
    annotations_train['categories'] = annotations['categories']
    annotations_val['categories'] = annotations['categories']

    # скопируем в валидацию каждое 10 изображение, а остальные - в обучение
    annotations_val['images'] = []
    annotations_train['images'] = []
    for num, img in enumerate(annotations['images']):
        if num % 10 == 0:
            annotations_val['images'].append(img)
        else:
            annotations_train['images'].append(img)

    # Положим в список аннотаций валидации только те аннотации, которые относятся к изображениям из валидации.
    # А в список аннотаций обучения - только те, которые относятся к нему
    val_img_id = [i['id'] for i in annotations_val['images']]
    train_img_id = [i['id'] for i in annotations_train['images']]

    annotations_val['annotations'] = []
    annotations_train['annotations'] = []

    for annot in annotations['annotations']:
        if annot['image_id'] in val_img_id:
            annotations_val['annotations'].append(annot)
        elif annot['image_id'] in train_img_id:
            annotations_train['annotations'].append(annot)
        else:
            print('Аннотации нет ни в одном наборе')

    return annotations_train, annotations_val


def copy_images(coco_obj: dict, source_dir: str, dest_dir: str):
    for image in tqdm(coco_obj['images']):
        os.makedirs(os.path.dirname(dest_dir + image['file_name']), exist_ok=True)
        shutil.copy(source_dir + image['file_name'], dest_dir + image['file_name'])

        
def prepare_masks(coco_filename: str, base_image_dir: str):
    # зачитываем coco файл
    with open(coco_filename, 'r')  as f:
        coco_data = json.load(f)

    # перебираем образы из coco файла и создаем по ним маски
    binary_mask_by_filenames = {}
    for coco_image in tqdm(coco_data["images"]):
        file_name = coco_image["file_name"]
        image = cv2.imread(os.path.join(base_image_dir, file_name))

        masks = []
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == coco_image["id"]:
                segmentation = annotation["segmentation"]

                if len(segmentation) > 0:
                    for segment in segmentation:
                        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

                        points = np.array(segment).reshape((-1, 2)).astype(np.int32)

                        # Отрисовка контура аннотации
                        npoints = len(points)

                        if npoints > 0:
                            cv2.drawContours(mask, [points], 0, 255, -1)
                            masks.append(mask)

        if len(masks) > 0:
            binary_mask = masks[0]
            for i in range(1, len(masks)):
                binary_mask = cv2.bitwise_or(binary_mask, masks[i])

            binary_mask = binary_mask > 0

            binary_mask_by_filenames[file_name] = binary_mask    
            
    return binary_mask_by_filenames
