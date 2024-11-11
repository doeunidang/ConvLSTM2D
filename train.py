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

X_train, y_train, X_val, y_val = load_train_val_data()

model = build_convlstm_model(input_shape=(4, 64, 64, 5))  # 입력 채널 5

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

checkpoint = ModelCheckpoint(
    '/content/ConvLSTM2D/model/convlstm_model.keras',
    monitor='val_loss', 
    save_best_only=True
)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    callbacks=[checkpoint, early_stopping]
)
