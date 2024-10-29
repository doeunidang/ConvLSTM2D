import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model
from utils import load_train_val_data

# 데이터 로드
X_train, y_train, X_val, y_val = load_train_val_data()

# 모델 빌드
model = build_convlstm_model()

# 콜백 설정
checkpoint = ModelCheckpoint('/content/ConvLSTM2D/convlstm_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 학습
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8, callbacks=[checkpoint, early_stopping])
