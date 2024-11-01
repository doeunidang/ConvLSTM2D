import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model
from utils import load_train_val_data
from losses import custom_loss
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 학습 및 검증 데이터 로드
X_train, y_train, X_val, y_val = load_train_val_data()

# ConvLSTM 모델 생성
model = build_convlstm_model(input_shape=(4, 64, 64, 9))

# Adam 옵티마이저에 학습률 설정
optimizer = Adam(learning_rate=0.0001)

# 모델 컴파일
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

# 콜백: 최적 모델 저장 및 조기 종료
checkpoint = ModelCheckpoint(
    '/content/ConvLSTM2D/model/convlstm_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=50
)

# 모델 학습
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=100, 
    batch_size=16,
    callbacks=[checkpoint, early_stopping]
)
