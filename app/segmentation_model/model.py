import os

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


def __init_predict_config__(config_path: str, model_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(cfg_filename=config_path)
    cfg.MODEL.WEIGHTS = model_path
    # ВАЖНО увеличить это значение (стандартное равно 100). Так как на листе может быть много слов
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    return cfg


class Segmentor:
    def __init__(self, config_path: str, model_path: str, device='cpu'):
        cfg = __init_predict_config__(config_path, model_path)
        self.predictor = DefaultPredictor(cfg)

    def segment(self, image) -> dict:
        image_id = 1
        image_name = 'input.jpg'
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "word"}]
        }
        outputs = self.predictor(image)

        pred_masks = outputs['instances'].pred_masks.cpu().numpy()
        pred_boxes = outputs['instances'].pred_boxes
        scores = outputs["instances"].scores.cpu()
        for i in range(len(outputs["instances"])):
            binary_mask = pred_masks[i]

            # поиск контуров на бинарной маске
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentations = []
            for contour in contours:
                segmentations.append([round(s, 1) for s in contour.flatten().tolist()])

            bbox = pred_boxes[i].tensor.cpu().tolist()[0]
            bbox = (round(bbox[0]), round(bbox[1]), round(bbox[2] - bbox[0]), round(bbox[3] - bbox[1]))

            annotation = {
                "image_id": image_id,
                "id": len(coco_data["annotations"]),
                "category_id": 1,
                "score": scores[i].item(),
                "color": "#1fd4fa",
                "bbox": bbox,
                "segmentation": segmentations,
                "metadata": {},
            }

            coco_data["annotations"].append(annotation)

        image_info = {
            "id": image_id,
            "file_name": image_name,
            "width": pred_masks.shape[2],
            "height": pred_masks.shape[1]
        }
        coco_data["images"].append(image_info)

        return coco_data


def get_segmentor_instance(model_path: str, config_path: str, device: str = 'cpu') -> Segmentor:
    return Segmentor(config_path, model_path, device)
