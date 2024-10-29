from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam

# ConvLSTM2D 모델 정의
def build_convlstm_model():
    model = Sequential()
    
    # 첫 번째 ConvLSTM2D 레이어
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         input_shape=(4, 64, 64, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    # 두 번째 ConvLSTM2D 레이어
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    # 세 번째 ConvLSTM2D 레이어
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())
    
    # Conv2D로 변경
    model.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
    
    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model
