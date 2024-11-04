from model import build_convlstm_model
from utils import load_train_val_data
from losses import custom_loss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# GPU 메모리 점진적 할당 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# 데이터 로드
X_train, y_train, X_val, y_val = load_train_val_data()

# ConvLSTM 모델 생성
model = build_convlstm_model(input_shape=(4, 64, 64, 5))  # 입력 채널 5

# 옵티마이저와 모델 컴파일
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

# 콜백 설정
checkpoint = ModelCheckpoint(
    '/content/ConvLSTM2D/model/convlstm_model.keras',
    monitor='val_loss', 
    save_best_only=True
)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,  # 배치 크기 줄임 (OOM 오류 방지용)
    callbacks=[checkpoint, early_stopping]
)
