"""Utilities for SSD

"""
import tensorflow as tf
import math
import itertools


def compute_area(top_left, bot_right):
    """Calculate the area of boxes with top left and bottom right coordinates

    Args:
        top_left: tensor (num_boxes, 2) -> (x, y)
        bot_right: tensor (num_boxes, 2) -> (x, y)

    Returns:
        area: tensor (num_boxes, )

    """
    height_width = bot_right - top_left
    height_width = tf.where(height_width < 0.0, 0.0, height_width)
    area = height_width[..., 0] * height_width[..., 1]

    return area


def compute_iou(boxes_1, boxes_2):
    """Calculate the value of intersection over union of paris of boxes. The
    boxes should be in format with coordinates of top left and bottom right.
    NOT the coordinates of center

    Args:
        boxes_1: tensor (num_boxes_1, 4) in format: (xmin, ymin, xmax, ymax)
        boxes_2: tensor (num_boxes_2, 4) in format: (xmin, ymin, xmax, ymax)

    Returns:
        ious: tensor (num_boxes_1, num_boxes_2)
    """
    # shape of boxes_1 => num_boxes_1, 1, 4
    boxes_1 = tf.expand_dims(boxes_1, 1)

    # shape of boxes_2 => 1, num_boxes_2, 4
    boxes_2 = tf.expand_dims(boxes_2, 0)
    top_left = tf.math.maximum(boxes_1[..., :2], boxes_2[..., :2])
    bot_right = tf.math.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

    intersection = compute_area(top_left, bot_right)
    area_1 = compute_area(boxes_1[..., :2], boxes_1[..., 2:])
    area_2 = compute_area(boxes_2[..., :2], boxes_2[..., 2:])

    ious = intersection / (area_1 + area_2 - intersection + tf.pow(10.0, -6))

    return ious


def transform_corner_to_center(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box


def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box


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


def generate_labels_from_annotation(default_boxes, gt_boxes, gt_labels,
                                    iou_threshold=0.5):
    """Generate training labels for SSD with default boxes and ground true

    Args:
        default_boxes: tensor (num_default, 4) of format (cx, cy, w, h)
        gt_boxes: tensor (num_gt_box, 4) of format (xmin, ymin, xmax, ymax)
        gt_labels: tensor (num_gt_box, )
        iou_threshold:

    Returns:
        gt_confs: classification targets, tensor (num_default,)
        gt_locs: regression targets, tensor (num_default, 4)
    """
    default_corner_box = transform_center_to_corner(default_boxes)
    iou = compute_iou(default_corner_box, gt_boxes)

    # match ground truth to default box, find the best gt_box for each
    # default box
    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    # match default box to ground truth
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    # assign label to each default box
    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)

    # assign gt boxes to each default box
    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs


def encode(default_boxes, boxes):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cx, cy, w, h) format
    transformed_boxes = transform_corner_to_center(boxes)

    locs = tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / default_boxes[:, 2:],
        tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:])],
        axis=-1)

    return locs


def main():
    cfg_path = "config.yaml"
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)
    default_box = generate_default_boxes(cfg["SSD300"])


if __name__ == '__main__':
    main()
