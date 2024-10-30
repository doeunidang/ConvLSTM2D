# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model
from utils import load_train_val_data, expand_mask_to_batch_size

# 데이터 로드
X_train, y_train, X_val, y_val, train_masks, val_masks = load_train_val_data()

# 마스크된 MSE 손실 함수
def masked_mse(y_true, y_pred):
    mask = expand_mask_to_batch_size(train_masks[0], tf.shape(y_true)[0])
    mask = tf.cast(mask, dtype=tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference * mask) / tf.reduce_sum(mask)

# 모델 빌드 및 컴파일
model = build_convlstm_model()
model.compile(
    optimizer='adam',
    loss=masked_mse,
    metrics=['mae']
)

# 콜백 설정
checkpoint = ModelCheckpoint('/content/ConvLSTM2D/convlstm_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 모델 학습
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=10, batch_size=8, 
                    callbacks=[checkpoint, early_stopping])
