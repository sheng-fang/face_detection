"""Functions for SSD LOSS

"""
import tensorflow as tf


def create_ssd_loss(nb_class):
    def ssd_loss(gt_confs, gt_locs, confs, locs):
        if nb_class == 1:
            cross_entropy = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction='none')
        else:
            cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

        temp_loss = cross_entropy(
            gt_confs, confs)

    return ssd_loss
