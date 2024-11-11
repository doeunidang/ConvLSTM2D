import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    junction_mask = np.load('/content/ConvLSTM2D/DATA_numpy/junction_mask.npy')
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    expanded_mask = tf.expand_dims(junction_mask, axis=0)
    expanded_mask = tf.expand_dims(expanded_mask, axis=0)
    expanded_mask = tf.expand_dims(expanded_mask, axis=-1)
    mask = tf.tile(expanded_mask, [tf.shape(y_true)[0], tf.shape(y_true)[1], 1, 1, 1])

    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask

    squared_difference = tf.square(masked_y_true - masked_y_pred)
    masked_squared_difference = squared_difference * mask  

    return tf.reduce_sum(masked_squared_difference) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
