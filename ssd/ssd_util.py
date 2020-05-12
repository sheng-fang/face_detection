"""Utilities for SSD

"""
import tensorflow as tf
import math
import itertools

from fslib import bbox_tf


def generate_default_boxes(config):
    """ Generate default boxes for all feature maps
    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps
    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    default_boxes = []
    scales = config['scales']
    fm_sizes = config['fm_sizes']
    ratios = config['ratios']

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes


def generate_labels_from_annotation(default_boxes, gt_boxes,
                                    iou_threshold=0.5):
    """Generate training labels for SSD with default boxes and ground true

    Args:
        default_boxes: tensor (num_default, 4) of format (cx, cy, w, h)
        gt_boxes: tensor (num_gt_box, 4) of format (xmin, ymin, xmax, ymax)
        iou_threshold:

    Returns:

    """
    default_corner_box = bbox_tf.transform_center_to_corner(default_boxes)
    iou = bbox_tf.compute_iou(default_corner_box, gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    # gt_confs = tf.gather(gt_labels, best_gt_idx)
    # gt_confs = tf.where(
    #     tf.less(best_gt_iou, iou_threshold),
    #     tf.zeros_like(gt_confs),
    #     gt_confs)
    #
    # gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    # gt_locs = encode(default_boxes, gt_boxes)
    #
    # return gt_confs, gt_locs


def main():
    cfg_path = "config.yaml"
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)
    default_box = generate_default_boxes(cfg["SSD300"])


if __name__ == '__main__':
    main()
