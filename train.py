from model import build_convlstm_model
from utils import load_train_val_data
from losses import custom_loss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import os

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

# 모델 정의
model = build_convlstm_model(input_shape=(4, 64, 64, 5))

# 모델 컴파일
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae', 'accuracy'])  # 'accuracy' 추가

# 콜백 설정
checkpoint = ModelCheckpoint(
    '/content/ConvLSTM2D/model/convlstm_model.keras',
    monitor='val_loss',
    save_best_only=True
)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint, early_stopping]
)

# 학습 시각화 저장
def plot_training_history(history, save_path_loss, save_path_accuracy):
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(save_path_loss), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_accuracy), exist_ok=True)

    # Loss 및 MAE 그래프
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path_loss)
    print(f"Training loss and MAE history saved at {save_path_loss}")
    plt.close()

    # Accuracy 그래프
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_accuracy)
        print(f"Training accuracy history saved at {save_path_accuracy}")
        plt.close()
    else:
        print("Accuracy metrics not available in the history.")

plot_training_history(
    history,
    '/content/ConvLSTM2D/outputs/training_history_loss_mae.png',
    '/content/ConvLSTM2D/outputs/training_history_accuracy.png'
)
