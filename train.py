import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model
from utils import load_train_val_data

# 학습 및 검증 데이터 로드
X_train, y_train, X_val, y_val = load_train_val_data()

# ConvLSTM 모델을 빌드
model = build_convlstm_model()

# 체크포인트와 조기 종료 설정
checkpoint = ModelCheckpoint('F:\\ConLSTM2D_TEST\\model\\convlstm_model.h5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 모델 학습
history = model.fit(
    X_train, y_train,  # 훈련 데이터
    validation_data=(X_val, y_val),  # 검증 데이터
    epochs=50,  # 최대 50 epoch 동안 학습
    batch_size=8,  # 배치 크기
    callbacks=[checkpoint, early_stopping]  # 콜백 설정
)
