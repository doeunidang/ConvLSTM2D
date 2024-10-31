import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model  # ConvLSTM 모델 생성
from utils import load_train_val_data   # 학습/검증 데이터 로드
from losses import custom_loss          # 커스텀 손실 함수
import tensorflow as tf

X_train, y_train, X_val, y_val = load_train_val_data()
model = build_convlstm_model(input_shape=(4, 64, 64, 9))

model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])
checkpoint = ModelCheckpoint(
    '/content/ConvLSTM2D/model/convlstm_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20
)
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=50, 
    batch_size=8,
    callbacks=[checkpoint, early_stopping]
)
