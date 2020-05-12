"""Create the SSD net

Sheng FANG
2020-05-12
"""
import tensorflow as tf


def create_ssd(input_shape, nb_class):
    pred_dim6 = 6 * (nb_class + 4)
    pred_dim4 = 4 * (nb_class + 4)
    vgg = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone = tf.keras.Sequential(vgg.layers[0:14])
    tmp = backbone.output
    pred_1 = tf.keras.layers.Conv2D(pred_dim4, 3, padding="same")(tmp)

    tmp = tf.keras.layers.Conv2D(1024, 3, padding="same")(tmp)
    tmp = tf.keras.layers.MaxPool2D()(tmp)
    tmp = tf.keras.layers.Conv2D(1024, 1, padding="same")(tmp)
    pred_2 = tf.keras.layers.Conv2D(pred_dim6, 3, padding="same")(tmp)

    tmp = tf.keras.layers.Conv2D(256, 1, padding="same")(tmp)
    tmp = tf.keras.layers.Conv2D(512, 3, strides=2, padding="same")(tmp)
    pred_3 = tf.keras.layers.Conv2D(pred_dim6, 3, padding="same")(tmp)

    tmp = tf.keras.layers.Conv2D(128, 1, padding="same")(tmp)
    tmp = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same")(tmp)
    pred_4 = tf.keras.layers.Conv2D(pred_dim6, 3, padding="same")(tmp)

    tmp = tf.keras.layers.Conv2D(128, 1, padding="same")(tmp)
    tmp = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same")(tmp)
    pred_5 = tf.keras.layers.Conv2D(pred_dim4, 3, padding="same")(tmp)

    tmp = tf.keras.layers.Conv2D(128, 1, padding="same")(tmp)
    tmp = tf.keras.layers.Conv2D(256, 3, strides=2, padding="valid")(tmp)
    pred_6 = tf.keras.layers.Conv2D(pred_dim4, 1, padding="valid")(tmp)

    ssd = tf.keras.Model(
        inputs=[backbone.input],
        outputs=[pred_1, pred_2, pred_3, pred_4, pred_5, pred_6],
    )

    return ssd


def main():
    input_shape = (304, 304, 3)
    ssd = create_ssd(input_shape, 20)
    ssd.summary()
    tf.keras.utils.plot_model(
        ssd, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )


if __name__ == '__main__':
    main()
