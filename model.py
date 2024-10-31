from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam

# ConvLSTM2D 기반의 모델을 생성하는 함수
def build_convlstm_model(input_shape=(4, 64, 64, 1)):
    model = Sequential()  # Sequential 모델 초기화
    # 첫 번째 ConvLSTM2D 레이어 (시퀀스를 유지하도록 설정)
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         input_shape=input_shape, padding='same', return_sequences=True))
    model.add(BatchNormalization())  # BatchNormalization 추가

    # 두 번째 ConvLSTM2D 레이어
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())  # BatchNormalization 추가

    # 세 번째 ConvLSTM2D 레이어 (시퀀스를 유지하지 않도록 설정)
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())  # BatchNormalization 추가

    # Conv2D 레이어 (출력 채널을 1로 설정, 최종 출력 레이어)
    model.add(Conv2D(filters=1, kernel_size=(3, 3), activation=None, padding='same'))

    # 모델 컴파일 (Adam 최적화 및 손실 함수: MSE, 평가 지표: MAE)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model
