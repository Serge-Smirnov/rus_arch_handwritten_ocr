import json
import os
import random
import shutil
from datetime import datetime
from datetime import timedelta

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import tqdm
# from IPython.display import clear_output

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase

mb_ds_train = 'metric_book_dataset_train'
mb_ds_val = 'metric_book_dataset_val'


def init_datasets(train_annotations_filename: str,
                  train_image_dir: str,
                  val_annotations_filename: str,
                  val_image_dir: str):
    try:
        DatasetCatalog.register(mb_ds_train,
                                lambda: load_coco_json(train_annotations_filename,
                                                       image_root=train_image_dir,
                                                       dataset_name=mb_ds_train))
        DatasetCatalog.register(mb_ds_val,
                                lambda: load_coco_json(val_annotations_filename,
                                                       image_root=val_image_dir,
                                                       dataset_name=mb_ds_val))
    except AssertionError:
        print("datasets already registered")

    print('Размер обучающей выборки (Картинки): {}'.format(len(DatasetCatalog.get(mb_ds_train))))
    print('Размер тестовой выборки (Картинки): {}'.format(len(DatasetCatalog.get(mb_ds_val))))


def init_train_config(config_filename: str, output_dir: str):
    # Создаем конфигурацию и загружаем архитектуру модели с предобученными весами (на COCO - датасете,
    # содержащем 80 популярных категорий объектов и более  300000 изображений) для распознавания объектов.
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # подгрузим наши кастомные настройки:
    cfg.merge_from_file(cfg_filename=config_filename)

    # И указываем название папки, куда сохранять чекпойнты модели и информацию о процессе обучения.
    cfg.OUTPUT_DIR = output_dir

    # Если вдруг такой папки нет, то создадим ее
    os.makedirs(output_dir, exist_ok=True)
    return cfg


class LossPlotter(HookBase):
    def __init__(self, trainer, period):
        self.trainer = trainer
        self.period = period
        self.losses = []

    def plot(self):
        # clear_output(wait=True)
        losses, iterations = zip(*self.losses)
        # plt.plot(iterations, losses)
        # plt.xlabel("Iterations")
        # plt.ylabel("total loss")
        # plt.title("Loss Curve")
        # plt.grid(True)
        # plt.show()

    def after_step(self):
        iteration = self.trainer.iter
        loss_dict = self.trainer.storage.latest()
        if iteration % self.period == 0:
            # Сохраняем значение потерь
            self.losses.append(loss_dict["total_loss"])
            if len(self.losses) > 2:
                self.plot()
                print(loss_dict)
                if "eta_seconds" in loss_dict.keys():
                    td = timedelta(seconds=loss_dict["eta_seconds"][0])
                    print('Осталось до окончания ', td)


def train_model(train_cfg: dict):
    print("всего итераций " + str(train_cfg.SOLVER.MAX_ITER))
    trainer = DefaultTrainer(train_cfg)
    trainer.resume_or_load(resume=False)
    loss_plotter = LossPlotter(train_cfg, period=20)
    trainer.register_hooks([loss_plotter])
    trainer.train()
    print("Обучение завершено")


def init_predict_config(config_filename: str, output_dir: str, model_name: str = "model_final.pth"):
    cfg = init_train_config(config_filename, output_dir)
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, model_name)
    # ВАЖНО увеличить это значение (стандартное равно 100). Так как на листе может быть много слов
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    return cfg


def predict(predict_cfg: dict, val_annotations_filename: str, image_dir: str):
    predictor = DefaultPredictor(predict_cfg)
    with open(val_annotations_filename, "r") as f:
        val_annotaions = json.load(f)
    val_images = val_annotaions['images']
    val_predictions = {}
    for val_img in tqdm.tqdm(val_images):
        file_name = val_img['file_name']
        img_path = os.path.join(image_dir, file_name)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        prediction = outputs['instances'].pred_masks.cpu().numpy()
        mask = np.add.reduce(prediction)
        mask = mask > 0
        val_predictions[file_name] = mask
    return val_predictions


def predict_image(predict_cfg: dict, image_file: str, output_coco_file: str):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "word"}]  # Замените "object_category" на реальное имя категории.
    }

    predictor = DefaultPredictor(predict_cfg)
    im = cv2.imread(image_file)
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

        # color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        bbox = pred_boxes[i].tensor.cpu().tolist()[0]

        bbox = (round(bbox[0]), round(bbox[1]), round(bbox[2] - bbox[0]), round(bbox[3] - bbox[1]))

        annotation = {
            "image_id": 1,
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
        "id": 1,  # Замените на реальный идентификатор изображения.
        "file_name": os.path.basename(image_file),  # Замените на реальный путь к изображению.
        "width": pred_masks.shape[2],
        "height": pred_masks.shape[1]
    }
    coco_data["images"].append(image_info)

    with open(output_coco_file, "w") as json_file:
        json.dump(coco_data, json_file)


def f1_loss(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return f1


def evaluate_masks_from_dict(groundtruth_masks: dict, predicted_masks: dict):
    f1_scores = []
    for key in tqdm.tqdm(predicted_masks.keys()):
        pred = predicted_masks[key].reshape(-1)
        true = groundtruth_masks[key].reshape(-1)

        f1_img = f1_loss(true, pred)
        f1_scores.append(f1_img)
    return round(np.mean(f1_scores), 5)


def evaluate_masks_from_npz_files(train_binary_mask_filename: str, val_pred_binary_mask_filename: str):
    loaded_train = np.load(train_binary_mask_filename)
    loaded_val_pred = np.load(val_pred_binary_mask_filename)
    groundtruth_masks_dict = {key: value for key, value in loaded_train.items()}
    predicted_masks_dict = {key: value for key, value in loaded_val_pred.items()}
    return evaluate_masks_from_dict(groundtruth_masks_dict, predicted_masks_dict)


def save_results(output_dir: str, files: list):
    dir = datetime.now().strftime("version_%Y.%m.%d_%H.%M")
    result_dir = os.path.join(output_dir, dir)
    os.makedirs(result_dir)
    for file in files:
        print('copy file: ' + file)
        shutil.copyfile(file, os.path.join(result_dir, os.path.basename(file)))
