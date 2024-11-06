import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    # Junction 마스크 파일 로드
    junction_mask = np.load('/content/ConvLSTM2D/DATA_numpy/junction_mask.npy')
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)

    # y_true와 y_pred의 형상이 (batch, time_steps, height, width, channels)임을 가정하고,
    # 마스크 형상 맞추기 (batch와 time_steps 차원 확장)
    expanded_mask = tf.expand_dims(junction_mask, axis=0)  # (1, 64, 64)
    expanded_mask = tf.expand_dims(expanded_mask, axis=0)  # (1, 1, 64, 64)
    expanded_mask = tf.expand_dims(expanded_mask, axis=-1) # (1, 1, 64, 64, 1)
    mask = tf.tile(expanded_mask, [tf.shape(y_true)[0], tf.shape(y_true)[1], 1, 1, 1])  # batch, time_steps 맞춤

    # 마스크 적용하여 Junction 위치에서만 손실 계산
    mask = tf.cast(mask, dtype=tf.float32)
    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask

    # Junction 위치에서만 손실 계산
    squared_difference = tf.square(masked_y_true - masked_y_pred)
    masked_squared_difference = squared_difference * mask  # 마스크 적용

    # 손실을 Junction 위치에서만 계산
    return tf.reduce_sum(masked_squared_difference) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
