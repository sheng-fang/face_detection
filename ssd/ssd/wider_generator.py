"""Data generator for VOC database

Sheng FANG
2020-05-14
"""
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from ssd.ssd import ssd_util


class WIDERGenerator(object):
    def __init__(self, data_dir, default_boxes, resize_shape,
                 augmentation=None, shuffle=True):
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.anno_path = os.path.join(self.data_dir, 'wider_array_anno.npy')
        self.default_boxes = default_boxes
        self.resize_shape = resize_shape
        self.anno = np.load(self.anno_path, allow_pickle=True)
        if shuffle:
            np.random.shuffle(self.anno)
        self.size = self.anno.shape[0]

        if augmentation is None:
            self.augmentation = ['original']
        else:
            self.augmentation = augmentation + ['original']

    def __len__(self):
        return self.size

    def _load_image(self, idx):
        """ Method to read image from file
            then resize to (300, 300)
            then subtract by ImageNet's mean
            then convert to Tensor
        Args:
            idx: the index to get filename from self.ids
        Returns:
            img: tensor of shape (3, 300, 300)
        """
        filename = self.anno[idx, 0]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path)

        return img

    def _get_annotation(self, idx, orig_shape):
        """ Method to read annotation from file
            Boxes are normalized to image size
            Integer labels are increased by 1
        Args:
            idx: the index to get filename from self.ids
            orig_shape: image's original shape
        Returns:
            boxes: numpy array of shape (num_gt, 4)
            labels: numpy array of shape (num_gt,)
        """
        h, w = orig_shape
        face_anno = np.frombuffer(self.anno[idx, 1]).reshape(-1, 10)
        boxes = face_anno[:, 0: 4]
        boxes = np.hstack([boxes[:, 0: 2], boxes[:, 0: 2]+boxes[:, 2: 4]])
        boxes = boxes / np.asarray([w, h, w, h])
        labels = np.zeros((boxes.shape[0], 1), dtype=np.int64)

        return boxes, labels

    def generate(self):
        """ The __getitem__ method
            so that the object can be iterable
        Args:
            index: the index to get filename from self.ids
        Returns:
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """

        for idx in range(self.size):
            # img, orig_shape = self._get_image(index)
            filename = self.anno[idx, 0]
            img = self._load_image(idx)
            w, h = img.size
            boxes, labels = self._get_annotation(idx, (h, w))
            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            # augmentation_method = np.random.choice(self.augmentation)
            # if augmentation_method == 'patch':
            #     img, boxes, labels = random_patching(img, boxes, labels)
            # elif augmentation_method == 'flip':
            #     img, boxes, labels = horizontal_flip(img, boxes, labels)

            img = np.array(img.resize(
                self.resize_shape), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = ssd_util.generate_labels_from_annotation(
                self.default_boxes, boxes, labels)

            yield filename, img, gt_confs, gt_locs


def main():

    cfg_path = "config.yaml"
    data_dir = os.path.join(os.path.expanduser("~"),
                            "Data/WIDER_FACE/WIDER_train")
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)
    default_box = ssd_util.generate_default_boxes(cfg["SSD300"])

    wider_gen = WIDERGenerator(data_dir, default_box, (304, 304))
    train_gen = wider_gen.generate

    train_dataset = (tf.data.Dataset.from_generator(
        train_gen, (tf.string, tf.float32, tf.int64, tf.float32)).batch(10)
                     )


if __name__ == '__main__':
    main()
