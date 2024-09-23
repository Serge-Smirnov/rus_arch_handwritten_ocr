import unittest

import numpy as np
from numpy.testing import assert_array_equal

from stage_1.train_model.metrics import is_bboxes_cross, shift_segment, create_binary_mask, get_bbox_by_segments, \
    get_annotation_only_with_segments_and_fill_bboxes, calc_confusion_matrix_for_segments, \
    to_annotations_by_image_name_map, calc_metrics_for_one_image


class TestCrossBboxes(unittest.TestCase):
    def test_is_bboxes_cross_when_equals_bbox_should_return_true(self):
        bboxes_a = (0, 0, 10, 10)
        bboxes_b = (0, 0, 10, 10)
        self.assertEqual(is_bboxes_cross(bboxes_a, bboxes_b), True)

    def test_is_bboxes_cross_when_second_bbox_is_to_the_right_of_first_should_return_false(self):
        bboxes_a = (0, 0, 10, 10)
        bboxes_b = (10, 0, 20, 10)
        self.assertEqual(is_bboxes_cross(bboxes_a, bboxes_b), False)

    def test_is_bboxes_cross_when_second_bbox_is_to_the_left_of_first_should_return_false(self):
        bboxes_a = (10, 0, 20, 10)
        bboxes_b = (0, 0, 10, 10)
        self.assertEqual(is_bboxes_cross(bboxes_a, bboxes_b), False)

    def test_is_bboxes_cross_when_second_bbox_lower_than_first_should_return_false(self):
        bboxes_a = (0, 0, 10, 10)
        bboxes_b = (0, 10, 10, 20)
        self.assertEqual(is_bboxes_cross(bboxes_a, bboxes_b), False)

    def test_is_bboxes_cross_when_second_bbox_upper_than_first_should_return_false(self):
        bboxes_a = (0, 10, 10, 20)
        bboxes_b = (0, 0, 10, 10)
        self.assertEqual(is_bboxes_cross(bboxes_a, bboxes_b), False)


class TestSegments(unittest.TestCase):
    def test_shift_segment(self):
        #         |x  y| x   y| x  y|
        segment = [1, 2, 10, 2, 5, 7]
        self.assertEqual(shift_segment(segment, -1, -2), [0, 0, 9, 0, 4, 5])

    def test_create_binary_mask(self):
        #         |x  y| x  y| x  y|
        segment = [1, 0, 3, 0, 1, 2]
        assert_array_equal(create_binary_mask(segment, 4, 3), np.array([
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
        ]))

    def test_calc_confusion_matrix_for_segments(self):
        #         0 1 1 1      0 1 1 1        0 tp tp tp
        # predict 0 1 1 0   gt 0 1 1 1   =>   0 tp tp fp
        #         0 1 0 0      0 0 1 0        0 fn fp 0
        predict_annotation = {'segmentation': [[1, 0, 3, 0, 1, 2]], 'bbox': (1, 0, 3, 2)}
        gt_annotation = {'segmentation': [[1, 0, 3, 0, 2, 2]], 'bbox': (1, 0, 3, 2)}
        self.assertEqual(calc_confusion_matrix_for_segments(predict_annotation, gt_annotation),
                         {'tp': 5, 'fp': 1, 'fn': 2})


class TestAnnotations(unittest.TestCase):
    def test_get_bbox_by_segments(self):
        #         |x  y| x  y| x  y|
        segment = [1, 0, 3, 0, 2, 1]
        annotation = {
            'segmentation': [segment]
        }
        self.assertEqual(get_bbox_by_segments(annotation), (1, 0, 3, 1))

    def test_get_annotation_only_with_segments_and_fill_bboxes(self):
        #         |x  y|
        segment = [0, 0]
        annotations = [
            {'id': 1, 'segmentation': [segment]},
            {'id': 2, 'segmentation': [segment]},
            {'id': 3}, ]
        self.assertEqual(get_annotation_only_with_segments_and_fill_bboxes(annotations), [
            {'id': 1, 'segmentation': [segment], 'bbox': (0, 0, 0, 0)},
            {'id': 2, 'segmentation': [segment], 'bbox': (0, 0, 0, 0)}
        ])

    def test_to_annotations_by_image_name_map(self):
        segment = [1, 0, 3, 0, 2, 1]
        coco = {
            'images': [{'id': 1, 'file_name': '1.jpg'}, {'id': 2, 'file_name': '2.jpg'}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'segmentation': [segment]},
                {'id': 2, 'image_id': 1, 'segmentation': [segment]},
                {'id': 3, 'image_id': 2, 'segmentation': [segment]}
            ]
        }
        self.assertEqual(to_annotations_by_image_name_map(coco), {
            '1.jpg': [
                {'id': 1, 'image_id': 1, 'segmentation': [segment]},
                {'id': 2, 'image_id': 1, 'segmentation': [segment]}
            ],
            '2.jpg': [
                {'id': 3, 'image_id': 2, 'segmentation': [segment]}
            ]
        })

    def test_calc_metrics_for_one_image(self):
        segment = [1, 0, 3, 0, 2, 1]
        predict_annotations = [{'id': 1, 'image_id': 1, 'segmentation': [segment]}]
        gt_annotations = [{'id': 1, 'image_id': 1, 'segmentation': [segment]}]
        self.assertEqual(calc_metrics_for_one_image(predict_annotations, gt_annotations), {
            'predict_count': 1,
            'gt_count': 1,
            'tp': 1,
            'fp': 0,
            'fn': 0,
            'metric_for_cross_segments': [{
                'predict_ann_id': 1,
                'gt_ann_id': 1,
                'tp_pixels': 4,
                'fp_pixels': 0,
                'fn_pixels': 0,
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0
            }],
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0
        })


if __name__ == '__main__':
    unittest.main()
