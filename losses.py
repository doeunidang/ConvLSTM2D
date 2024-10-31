import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)
