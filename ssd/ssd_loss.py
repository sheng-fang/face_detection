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


def create_ssd_loss(nb_class, neg_ratio):
    """

    Args:
        nb_class:
        neg_ratio:

    Returns:

    """
    def ssd_loss(gt_confs, gt_locs, confs, locs):
        if nb_class == 1:
            cross_entropy = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, reduction='none')
        else:
            cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

        temp_loss = cross_entropy(
            gt_confs, confs)
        pos_idx, neg_idx = hard_negative_mining(
            temp_loss, gt_confs, neg_ratio)

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)])

        # regression loss only consist of positive examples
        loc_loss = smooth_l1_loss(
            # tf.boolean_mask(gt_locs, pos_idx),
            # tf.boolean_mask(locs, pos_idx))
            gt_locs[pos_idx],
            locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss

    return ssd_loss
