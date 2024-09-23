import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from stage_1.prepare_data.prepare_data import prepare_masks
from stage_1.train_model.training import evaluate_masks_from_dict


def __init_predict_config__(config_path: str, model_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(cfg_filename=config_path)
    cfg.MODEL.WEIGHTS = model_path
    # ВАЖНО увеличить это значение (стандартное равно 100). Так как на листе может быть много слов
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    return cfg


def predict_and_draw_segment_on_test_images(model_file_path: str, config_file_path: str, test_dir: str,
                                            annotation_file_name: str = "test_annotations.json"):
    """
        Проверка качества обученной модели на тестовой выборке и сохранением образов с детектированными сегментами
    """
    predictor = DefaultPredictor(__init_predict_config__(config_file_path, model_file_path))
    test_annotations_filename = os.path.join(test_dir, annotation_file_name)
    with open(test_annotations_filename, "r") as f:
        val_annotaions = json.load(f)
    test_images = val_annotaions['images']
    image_dir = os.path.join(test_dir, 'images')
    out_image_dir = os.path.join(test_dir, 'saved_segmentated_images')
    test_prediction_masks = {}
    for test_img in tqdm(test_images):
        try:
            # зачитываем картинку
            file_name = test_img['file_name']
            img_path = os.path.join(image_dir, file_name)
            im = cv2.imread(img_path)

            # предсказываем/детектируем сегменты
            outputs = predictor(im)

            # выводим все bbox в начало координат в (0,0) чтобы не мешались при отрисовке сегментов
            output_cpu = outputs["instances"].to("cpu")
            output_cpu.pred_boxes.tensor.fill_(0)

            # рисуем сегменты на картинке и сохраняем её
            v = Visualizer(im[:, :])
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_img_path = os.path.join(out_image_dir, file_name)
            cv2.imwrite(out_img_path, v.get_image()[:, :, ::-1])

            # добавляем бинарную маску сегментов
            prediction = outputs['instances'].pred_masks.cpu().numpy()
            mask = np.add.reduce(prediction)
            mask = mask > 0
            test_prediction_masks[file_name] = mask

        except Exception as e:
            print(e.__traceback__, e)

    test_groundtruth_masks = prepare_masks(test_annotations_filename, image_dir)
    f1 = evaluate_masks_from_dict(test_groundtruth_masks, test_prediction_masks)
    print(f'F1 = {f1}')


def predict_and_save_result_to_coco_format(model_file_path: str, config_file_path: str, test_dir: str,
                                           annotation_file_name: str = "test_annotations.json"):
    """
        Запуск модели на тестовой выборке с сохранением результатов в файл coco формата
    """
    predictor = DefaultPredictor(__init_predict_config__(config_file_path, model_file_path))
    test_annotations_filename = os.path.join(test_dir, annotation_file_name)
    with open(test_annotations_filename, "r") as f:
        val_annotaions = json.load(f)
    test_images = val_annotaions['images']
    image_dir = os.path.join(test_dir, 'images')

    val_annotaions["annotations"] = []
    for test_img in tqdm(test_images):
        try:
            # зачитываем картинку
            file_name = test_img['file_name']
            img_path = os.path.join(image_dir, file_name)
            im = cv2.imread(img_path)

            # предсказываем/детектируем сегменты
            outputs = predictor(im)
            pred_masks = outputs['instances'].pred_masks.cpu().numpy()
            pred_boxes = outputs['instances'].pred_boxes
            scores = outputs["instances"].scores.cpu()
            for i in range(len(outputs["instances"])):
                binary_mask = pred_masks[i]

                # Найдите контуры на бинарной маске
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentations = []
                for contour in contours:
                    segmentations.append([round(s, 1) for s in contour.flatten().tolist()])

                bbox = pred_boxes[i].tensor.cpu().tolist()[0]
                bbox = (round(bbox[0]), round(bbox[1]), round(bbox[2] - bbox[0]), round(bbox[3] - bbox[1]))

                annotation = {
                    "image_id": test_img['id'],
                    "id": len(val_annotaions["annotations"]),
                    "category_id": 1,
                    "score": scores[i].item(),
                    "bbox": bbox,
                    "segmentation": segmentations,
                }

                val_annotaions["annotations"].append(annotation)


        except Exception as e:
            print(e.__traceback__, e)
            raise e

    predict_annotations_filename = os.path.join(test_dir, "predict_annotations.json")
    with open(predict_annotations_filename, "w") as json_file:
        json.dump(val_annotaions, json_file)
