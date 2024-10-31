import tensorflow as tf

# 커스텀 손실 함수를 등록하여 Keras 모델에서 사용할 수 있도록 설정
@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    # y_true에서 NaN 값이 있는 위치를 마스킹하여 무시
    mask = tf.math.logical_not(tf.math.is_nan(y_true))  # NaN 위치를 False로 표시
    y_true = tf.boolean_mask(y_true, mask)  # NaN이 아닌 값만 남김
    y_pred = tf.boolean_mask(y_pred, mask)  # y_pred에서도 동일한 위치의 값만 남김
    
    # junction 위치에 대해서만 평균 제곱 오차(MSE)를 계산
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)
