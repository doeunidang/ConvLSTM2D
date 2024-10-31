from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam

def build_convlstm_model(input_shape=(4, 64, 64, 9)):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         input_shape=input_shape, padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(3, 3), activation=None, padding='same'))
    return model
