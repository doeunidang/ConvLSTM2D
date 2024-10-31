import tensorflow as tf

# 커스텀 손실 함수 정의
@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))  # NaN 위치를 무시
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    mse = tf.keras.losses.MeanSquaredError()  # MSE 계산
    return mse(y_true, y_pred)
