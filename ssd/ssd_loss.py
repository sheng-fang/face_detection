"""Functions for SSD LOSS

"""
import tensorflow as tf


def hard_negative_mining(loss, gt_confs, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B,num_default)
        gt_confs: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        conf_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # gt_confs: B x N
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


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
