import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model
from utils import load_train_val_data, load_train_val_masks

import tensorflow as tf

# 마스크를 사용하는 MSE 손실 함수
def masked_mse(y_true, y_pred, mask):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    squared_difference = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(squared_difference) / tf.reduce_sum(mask)

# 데이터 로드
X_train, y_train, X_val, y_val = load_train_val_data()
train_masks, val_masks = load_train_val_masks()

# 모델 빌드
model = build_convlstm_model()

# 컴파일 - lambda를 사용하여 train_masks 전달
model.compile(optimizer='adam', loss=lambda y_true, y_pred: masked_mse(y_true, y_pred, train_masks), metrics=['mae'])

# 콜백 설정
checkpoint = ModelCheckpoint('/content/ConvLSTM2D/convlstm_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 학습
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, callbacks=[checkpoint, early_stopping])
