import json

from prepare_data.prepare_data import get_coco_files, save_to_file, merge, fix_image_paths, \
    prepare_train_and_val_coco_files, save_to_file, copy_images, prepare_masks
from train_model.training import init_datasets, init_train_config, train_model
from train_model.training import init_predict_config, predict, predict_image
from train_model.training import evaluate_masks_from_npz_files
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger
import numpy as np

data_dir = './hcr-archive-metric-book/data_mb/'

coco_files_dir = data_dir + 'coco/metadata_for_merge/'
merged_coco_filename = data_dir + 'coco/merged.json'
image_dir = data_dir + 'images/'

# обучение
train_dir = data_dir + 'train/'
train_image_dir = train_dir + 'images/'
train_annotations_filename = train_dir + 'train_annotations.json'
train_binary_mask_filename = train_dir + 'all_groundtruth_binary_mask.npz'
train_config_filename = train_dir + "config.yaml"

# валидация
val_dir = data_dir + 'val/'
val_image_dir = val_dir + 'images/'
val_annotations_filename = val_dir + 'val_annotations.json'
val_binary_mask_filename = val_dir + 'val_binary_mask.npz'
val_predict_binary_mask_filename = val_dir + 'val_predict_binary_mask.npz'

output_dir = data_dir + '/stage-1/output/'
model_final_filename = output_dir + 'model_final.pth'
metrics_filename = output_dir + 'metrics.json'


def merge_coco_files():
    coco_files = get_coco_files(coco_files_dir)
    merged_coco_obj = merge(coco_files)
    fix_image_paths(merged_coco_obj, image_dir)
    save_to_file(merged_coco_obj, merged_coco_filename)


def validate_coco_file():
    with open(merged_coco_filename, "r") as f:
        coco_data = json.load(f)

    for annotation in coco_data["annotations"]:
        segment_count = len(annotation["segmentation"])
        if segment_count > 1:
            print("more one segment - " + str(annotation["image_id"]))
            print(annotation)


def prepare_datasets():
    annotations_train, annotations_val = prepare_train_and_val_coco_files(merged_coco_filename)

    save_to_file(annotations_train, train_annotations_filename)
    save_to_file(annotations_val, val_annotations_filename)

    # Скопируем изображения для обучения и валидации
    copy_images(annotations_train, image_dir, train_image_dir)
    copy_images(annotations_val, image_dir, val_image_dir)


def prepare_binary_masks():
    # сохраняем маски в файл
    all_binary_mask_dict = prepare_masks(merged_coco_filename, image_dir)
    np.savez_compressed(train_binary_mask_filename, **all_binary_mask_dict)

    val_binary_mask_dict = prepare_masks(val_annotations_filename, val_image_dir)
    np.savez_compressed(val_binary_mask_filename, **val_binary_mask_dict)


def train():
    setup_logger()

    init_datasets(
        train_annotations_filename=train_annotations_filename,
        train_image_dir=train_image_dir,
        val_annotations_filename=val_annotations_filename,
        val_image_dir=val_image_dir)

    cfg = init_train_config(train_config_filename, output_dir)

    # обучение
    # train_model(cfg)
    # или:
    launch(train_model, num_gpus_per_machine=1, args=(cfg,))
    # не работает multy gpu: launch(train_model, num_gpus_per_machine=2, args=(cfg,))


def validate():
    # построим предсказания на валидационном датасете и сохраним бинарные маски в файл
    cfg = init_predict_config(train_config_filename, output_dir)
    val_predictions_binary_mask_dict = predict(cfg, val_annotations_filename, val_image_dir)
    np.savez_compressed(val_predict_binary_mask_filename, **val_predictions_binary_mask_dict)

    # посчитаем метрику F1-score на валидационном датасете
    f1 = evaluate_masks_from_npz_files(train_binary_mask_filename, val_predict_binary_mask_filename)
    print(f1)


if __name__ == '__main__':
    # merge_coco_files()
    validate_coco_file()
    # prepare_datasets()
    # prepare_binary_masks()
    # train()
    # validate()

    # cfg = init_predict_config(train_config_filename, output_dir)
    # predict_image(cfg, "./hcr-archive-metric-book/data_mb/val/images/19-127-125-595039.jpg",
    #                      "./hcr-archive-metric-book/data_mb/val/images/19-127-125-595039.json")

