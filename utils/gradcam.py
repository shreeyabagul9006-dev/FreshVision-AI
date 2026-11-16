
import numpy as np
import tensorflow as tf
import cv2


def generate_gradcam(model, image_array, class_index):

    base_model = model.layers[1]
    target_layer = base_model.get_layer("out_relu")

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=target_layer.output
    )

    with tf.GradientTape() as tape:

        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        preprocessed = model.layers[0](image_tensor)

        conv_outputs = feature_extractor(
            preprocessed,
            training=False
        )

        tape.watch(conv_outputs)

        predictions = conv_outputs

        for layer in model.layers[2:]:
            predictions = layer(predictions, training=False)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap[0].numpy()

    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (160, 160))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap
