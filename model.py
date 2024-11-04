from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.optimizers import Adam

# ConvLSTM 모델 빌드 함수
def build_convlstm_model(input_shape=(4, 64, 64, 5)):
    """
    ConvLSTM2D 기반 모델을 생성합니다.
    input_shape: (타임스텝, 높이, 너비, 채널 수)
    """
    model = Sequential()
    
    # ConvLSTM2D 레이어 추가
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         input_shape=input_shape, padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    # 여러 타임스텝을 예측하도록 Conv3D 레이어 사용
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    
    return model
