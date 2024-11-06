import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Multiply, Input
from tensorflow.keras.optimizers import Adam

def build_convlstm_model(input_shape=(4, 64, 64, 5), junction_mask_path='/content/ConvLSTM2D/DATA_numpy/junction_mask.npy'):
    """
    ConvLSTM2D 기반 모델을 생성하고 Junction 위치에서만 예측이 활성화되도록 마스킹 적용.
    input_shape: (타임스텝, 높이, 너비, 채널 수)
    """
    # Junction 마스크 불러오기
    junction_mask = np.load(junction_mask_path)
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    
    # 차원 확장을 한 번에 한 개씩 수행하여 최종 모양을 맞춤
    junction_mask = tf.expand_dims(junction_mask, axis=0)  # (1, 64, 64)
    junction_mask = tf.expand_dims(junction_mask, axis=0)  # (1, 1, 64, 64)
    junction_mask = tf.expand_dims(junction_mask, axis=-1) # (1, 1, 64, 64, 1)로 최종 확장
    
    # 모델 입력 정의
    inputs = Input(shape=input_shape)
    
    # ConvLSTM2D 레이어 추가
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    
    # Conv3D를 사용해 다수의 타임스텝을 예측
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    
    # 마스킹 처리: Junction 위치에서만 예측 결과를 활성화
    x = Multiply()([x, junction_mask])  # `Multiply` 레이어로 마스킹 적용
    
    model = Model(inputs=inputs, outputs=x)
    return model
