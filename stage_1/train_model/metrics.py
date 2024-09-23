import json

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def calc_metrics_all(predict_coco_annotations: dict, gt_coco_annotations: dict):
    image_name_to_gt_annotations_map = to_annotations_by_image_name_map(gt_coco_annotations)
    image_name_to_predict_annotations_map = to_annotations_by_image_name_map(predict_coco_annotations)

    metrics = []
    for image_name in tqdm(image_name_to_predict_annotations_map.keys()):
        metrics.append({
            "image_id": image_name,
            "metrics": calc_metrics_for_one_image(
                predict_coco_annotations=image_name_to_predict_annotations_map[image_name],
                gt_coco_annotations=image_name_to_gt_annotations_map[image_name])}
        )
    return metrics


def to_annotations_by_image_name_map(coco_annotations):
    image_name_to_annotations_map = {}
    for image in coco_annotations["images"]:
        annotations = [a for a in coco_annotations["annotations"] if a["image_id"] == image["id"]]
        image_name_to_annotations_map[image["file_name"]] = annotations
    return image_name_to_annotations_map


def calc_metrics_for_one_image(predict_coco_annotations: list, gt_coco_annotations: list):
    gt_bboxes: list = get_annotation_only_with_segments_and_fill_bboxes(gt_coco_annotations)
    predict_bboxes: list = get_annotation_only_with_segments_and_fill_bboxes(predict_coco_annotations)
    count_cross_segments = 0
    metrics_for_cross_segments = []
    for predict_ann in predict_bboxes:
        max_precision = 0
        metric_for_cross_segments = None
        for gt_ann in gt_bboxes:
            if is_bboxes_cross(predict_ann["bbox"], gt_ann["bbox"]):
                confusion_matrix = calc_confusion_matrix_for_segments(predict_ann, gt_ann)
                precision = calc_precision(confusion_matrix)
                if precision > 0.25 and precision > max_precision:
                    max_precision = precision
                    metric_for_cross_segments = {
                        "predict_ann_id": predict_ann["id"],
                        "gt_ann_id": gt_ann["id"],
                        "tp_pixels": int(confusion_matrix["tp"]),
                        "fp_pixels": int(confusion_matrix["fp"]),
                        "fn_pixels": int(confusion_matrix["fn"]),
                        "precision": precision,
                        "recall": calc_recall(confusion_matrix),
                        "f1": calc_f1_score(confusion_matrix)
                    }
        if metric_for_cross_segments is not None:
            metrics_for_cross_segments.append(metric_for_cross_segments)
            count_cross_segments += 1

    result = {
        "predict_count": len(predict_bboxes),
        "gt_count": len(gt_bboxes),
        "tp": count_cross_segments,
        "fp": len(gt_bboxes) - count_cross_segments,
        "fn": len(predict_bboxes) - count_cross_segments,
        "metric_for_cross_segments": metrics_for_cross_segments
    }
    result["precision"] = calc_precision(result)
    result["recall"] = calc_recall(result)
    result["f1"] = calc_f1_score(result)
    return result


def get_annotation_only_with_segments_and_fill_bboxes(image_coco_annotations: list) -> list:
    result = []
    for annotation in image_coco_annotations:
        if "segmentation" in annotation:
            result.append({
                "id": annotation["id"],
                "segmentation": annotation["segmentation"],
                "bbox": get_bbox_by_segments(annotation)
            })
    return result


def get_bbox_by_segments(annotation: dict):
    """
        return tuple (min_x, min_y, max_x, max_y)
    """
    segment = np.array(annotation["segmentation"][0])
    x_coords = segment[::2]
    y_coords = segment[1::2]
    min_x, max_x = np.amin(x_coords), np.amax(x_coords)
    min_y, max_y = np.amin(y_coords), np.amax(y_coords)
    return min_x, min_y, max_x, max_y


def is_bboxes_cross(bboxes_a: tuple, bboxes_b: tuple) -> bool:
    a_min_x, a_min_y, a_max_x, a_max_y = bboxes_a
    b_min_x, b_min_y, b_max_x, b_max_y = bboxes_b
    #  +───────────────────────────────────────────────────────────────►X
    #  │  (a_min_x, a_min_y) +─────────────┐
    #  │                     │             │
    #  │                     └─────────────+ (a_max_x, a_max_y)
    #  │
    #  │          (b_min_x, b_min_y) +─────────────┐
    #  │                             │             │
    #  │                             └─────────────+ (b_max_x, b_max_y)
    # Y▼
    if a_max_x <= b_min_x or a_min_x >= b_max_x or a_max_y <= b_min_y or a_min_y >= b_max_y:
        return False
    return True


def calc_confusion_matrix_for_segments(predict_annotation: dict, gt_annotation: dict) -> dict:
    """
        predict_annotation: predicted annotation, must contain bbox and segment
        gt_annotation: groundtruth annotation, must contain bbox and segment
        return confusion matrix {tp, fp, fn} for segments
    """
    a_min_x, a_min_y, a_max_x, a_max_y = predict_annotation["bbox"]
    b_min_x, b_min_y, b_max_x, b_max_y = gt_annotation["bbox"]
    width = int(max(a_max_x, b_max_x) - min(a_min_x, b_min_x)) + 1
    height = int(max(a_max_y, b_max_y) - min(a_min_y, b_min_y)) + 1
    shift_x = min(a_min_x, b_min_x)
    shift_y = min(a_min_y, b_min_y)
    predict_segment = shift_segment(predict_annotation["segmentation"][0], -shift_x, -shift_y)
    gt_segment = shift_segment(gt_annotation["segmentation"][0], -shift_x, -shift_y)
    predict_mask = create_binary_mask(predict_segment, width, height).reshape(-1)
    gt_mask = create_binary_mask(gt_segment, width, height).reshape(-1)

    tp = np.sum(predict_mask & gt_mask)
    fp = np.sum(predict_mask & ~gt_mask)
    fn = np.sum(~predict_mask & gt_mask)

    return {"tp": tp, "fp": fp, "fn": fn}


def shift_segment(segment, dx, dy):
    return [round(element + dx, 1) if i % 2 == 0 else round(element + dy, 1) for i, element in enumerate(segment)]


def create_binary_mask(segment, width: int, height: int):
    image_shape = (height, width)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # преобразование одномерного массива координат в двумерный массив, где каждая строка содержит пару координат (x, y):
    points = np.array(segment, np.int32).reshape((-1, 2))
    cv2.fillPoly(mask, [points], 1)
    return mask


def is_segments_cross(confusion_matrix):
    return calc_precision(confusion_matrix) > 0.25


def calc_f1_score(confusion_matrix: dict):
    precision = calc_precision(confusion_matrix)
    recall = calc_recall(confusion_matrix)
    epsilon = 1e-7
    return round(2 * precision * recall / (precision + recall + epsilon), 2)


def calc_precision(confusion_matrix: dict):
    tp = confusion_matrix["tp"]
    fp = confusion_matrix["fp"]
    epsilon = 1e-7
    return round(tp / (tp + fp + epsilon), 2)


def calc_recall(confusion_matrix: dict):
    tp = confusion_matrix["tp"]
    fn = confusion_matrix["fn"]
    epsilon = 1e-7
    return round(tp / (tp + fn + epsilon), 2)


def plot_metrics(all_metrics: list):
    image_numbers = []
    predict_counts = []
    gt_counts = []
    precisions = []
    recalls = []
    f1_scores = []

    precision_data = []
    recall_data = []
    f1_data = []

    for entry in all_metrics:
        image_number = len(image_numbers) + 1
        image_numbers.append(image_number)
        image_metrics = entry['metrics']
        predict_counts.append(image_metrics['predict_count'])
        gt_counts.append(image_metrics['gt_count'])
        precisions.append(image_metrics['precision'])
        recalls.append(image_metrics['recall'])
        f1_scores.append(image_metrics['f1'])

        for segment_metrics in image_metrics['metric_for_cross_segments']:
            precision_data.append(segment_metrics['precision'])
            recall_data.append(segment_metrics['recall'])
            f1_data.append(segment_metrics['f1'])

    ### for segments counts
    plt.figure(figsize=(12, 6))
    plt.plot(image_numbers, precisions,
             label='Precision - точность, '
                   'показывает долю верно предсказанных сегментов среди всех истинных сегментов объектов. '
                   'Среднее ' + str(round(np.mean(precisions), 3)))
    plt.plot(image_numbers, recalls,
             label='Recall - полнота, '
                   'показывает отношение верно предсказанных сегментов к общему числу предсказанных сегментов. '
                   'Среднее ' + str(round(np.mean(recalls), 3)))
    plt.xticks(rotation=90)
    plt.xlabel('Номер образа')
    plt.ylabel('')
    plt.title('Метрика по соотношению количества предсказанных сегментов к истинным.\n'
              '(Всего: предсказанных - ' + str(np.sum(predict_counts)) + ", истинных - " + str(np.sum(gt_counts)) + ")")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(image_numbers, f1_scores,
             label='F1 - F-мера, среднее гармоническое между precision и recall. '
                   'Среднее ' + str(round(np.mean(precisions), 3)))
    plt.xticks(rotation=90)
    plt.xlabel('Номер образа')
    plt.ylabel('')
    plt.title('Метрика по соотношению количества предсказанных сегментов к истинным.\n'
              '(Всего: предсказанных - ' + str(np.sum(predict_counts)) + ", истинных - " + str(np.sum(gt_counts)) + ")")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ### for segments areas
    plt.figure(figsize=(12, 6))
    plt.hist(precision_data, bins=50, color='red',
             label='Precision - точность, показывает долю площади предсказанной к истинной. '
                   'Среднее ' + str(round(np.mean(precision_data), 3)))
    plt.xlabel('значение')
    plt.ylabel('количество сегментов')
    plt.title('Распределение метрик (площадей предсказанных сегментов к истинным) по верно предсказанным сегментам')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(recall_data, bins=50, color='green',
             label='Recall - полнота, показывает отношение площади верно предсказанной ко всей предсказанной площади. '
                   'Среднее ' + str(round(np.mean(recall_data), 3)))
    plt.xlabel('значение')
    plt.ylabel('количество сегментов')
    plt.title('Распределение метрик (площадей предсказанных сегментов к истинным) по верно предсказанным сегментам')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(f1_data, bins=50, color='blue',
             label='F1 - F-мера, среднее гармоническое между precision и recall. '
                   'Среднее ' + str(round(np.mean(f1_data), 3)))
    plt.xlabel('значение')
    plt.ylabel('количество сегментов')
    plt.title('Распределение метрик (площадей предсказанных сегментов к истинным) по верно предсказанным сегментам')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    with open("../../data_mb/test/predict_annotations.json", "r") as f:
        predict_annotations = json.load(f)
    with open("../../data_mb/test/test_annotations.json", "r") as f:
        gt_annotations = json.load(f)

    res = calc_metrics_all(predict_coco_annotations=predict_annotations, gt_coco_annotations=gt_annotations)
    plot_metrics(res)
