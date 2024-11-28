import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from model import build_convlstm_model
from utils import load_train_val_data

# XLA JIT 및 MLIR 비활성화
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=0"
os.environ["TF_DISABLE_MLIR_GRAPH_OPTIMIZATION"] = "1"

# 그래프 모드 비활성화 (Eager Execution 활성화)
tf.config.run_functions_eagerly(True)

def load_junction_mask(junction_mask_path='/content/ConvLSTM2D/DATA_numpy/junction_mask.npy'):
    """
    Junction mask를 로드하고 배치/시간/채널 차원을 추가합니다.
    """
    junction_mask = np.load(junction_mask_path)  # (64, 64, 1)
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    junction_mask = tf.expand_dims(junction_mask, axis=0)  # 배치 차원 추가
    junction_mask = tf.expand_dims(junction_mask, axis=0)  # 시간 차원 추가
    junction_mask = tf.expand_dims(junction_mask, axis=-1)  # 채널 차원 추가
    return junction_mask  # (1, 1, 64, 64, 1)

@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_mae_loss")
def custom_loss(y_true, y_pred):
    junction_mask = np.load('/content/ConvLSTM2D/DATA_numpy/junction_mask.npy')
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    expanded_mask = tf.expand_dims(junction_mask, axis=0)
    expanded_mask = tf.expand_dims(expanded_mask, axis=0)
    expanded_mask = tf.expand_dims(expanded_mask, axis=-1)
    mask = tf.tile(expanded_mask, [tf.shape(y_true)[0], tf.shape(y_true)[1], 1, 1, 1])

    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask

    absolute_difference = tf.abs(masked_y_true - masked_y_pred)
    masked_absolute_difference = absolute_difference * mask  

    return tf.reduce_sum(masked_absolute_difference) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())


@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_accuracy")
def custom_accuracy(y_true, y_pred):
    """
    유출량 예측에서 상대 오차 또는 절대 오차를 기준으로 정확도를 계산합니다.
    """
    # Junction Mask 로드
    junction_mask = load_junction_mask()  # (1, 1, 64, 64, 1)
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    # Junction Mask를 y_true, y_pred 크기로 Broadcast
    junction_mask_broadcasted = tf.tile(junction_mask, [batch_size, time_steps, 1, 1, 1])

    # Masking
    y_true_masked = y_true * junction_mask_broadcasted
    y_pred_masked = y_pred * junction_mask_broadcasted

    # 상대 오차 및 절대 오차 계산
    relative_error = tf.abs(y_true_masked - y_pred_masked) / (y_true_masked + tf.keras.backend.epsilon())
    absolute_error = tf.abs(y_true_masked - y_pred_masked)

    # 허용 범위: 상대 오차 <= 20% 또는 절대 오차 <= 1.0
    within_tolerance = tf.logical_or(relative_error <= 0.2, absolute_error <= 0.5)

    # 정확한 예측 개수 계산
    correct_predictions = tf.reduce_sum(tf.cast(within_tolerance, tf.float32) * junction_mask_broadcasted, axis=[2, 3, 4])
    total_junctions = tf.reduce_sum(junction_mask_broadcasted, axis=[2, 3, 4])

    # Frame별 Accuracy 계산
    accuracy_per_frame = correct_predictions / (total_junctions + tf.keras.backend.epsilon())

    # Mean Accuracy 계산
    mean_accuracy = tf.reduce_mean(accuracy_per_frame)

    return mean_accuracy


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
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae', custom_accuracy])

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

def plot_training_history(history, save_path_loss, save_path_accuracy):
    """
    Training and validation performance plotting function.
    """
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
    if 'custom_accuracy' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['custom_accuracy'], label='Train Accuracy (Junction)')
        plt.plot(history.history['val_custom_accuracy'], label='Validation Accuracy (Junction)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy (Junction)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path_accuracy)
        print(f"Training accuracy history saved at {save_path_accuracy}")
        plt.close()

# 학습 시각화 저장
plot_training_history(
    history,
    '/content/ConvLSTM2D/outputs/training_history_loss_mae.png',
    '/content/ConvLSTM2D/outputs/training_history_accuracy.png'
)
