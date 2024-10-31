import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model  # ConvLSTM 모델을 생성하는 함수 임포트
from utils import load_train_val_data   # 학습 및 검증 데이터를 로드하는 함수 임포트
from losses import custom_loss          # 커스텀 손실 함수를 임포트
import tensorflow as tf

# 학습 및 검증 데이터 로드
X_train, y_train, X_val, y_val = load_train_val_data()

# ConvLSTM 모델 생성
model = build_convlstm_model(input_shape=(4, 64, 64, 1))

# 모델 컴파일 (최적화: Adam, 손실 함수: custom_loss, 평가 지표: MAE)
model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])

# 체크포인트 콜백 설정 (가장 낮은 검증 손실을 기준으로 모델 저장)
checkpoint = ModelCheckpoint(
    '/content/ConvLSTM2D/model/convlstm_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)

# EarlyStopping 콜백 설정 (검증 손실 기준으로 20회 동안 개선이 없을 경우 학습 중단)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20
)

# 모델 학습 시작 (학습 데이터와 검증 데이터, 총 50 에포크, 배치 크기 8, 콜백 포함)
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=50, 
    batch_size=8,
    callbacks=[checkpoint, early_stopping]
)
