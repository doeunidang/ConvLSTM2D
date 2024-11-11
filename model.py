import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Multiply, Input

def build_convlstm_model(input_shape=(4, 64, 64, 5), junction_mask_path='/content/ConvLSTM2D/DATA_numpy/junction_mask.npy'):
    junction_mask = np.load(junction_mask_path)
    junction_mask = tf.convert_to_tensor(junction_mask, dtype=tf.float32)
    junction_mask = tf.expand_dims(junction_mask, axis=0)
    junction_mask = tf.expand_dims(junction_mask, axis=0)
    junction_mask = tf.expand_dims(junction_mask, axis=-1)
    
    inputs = Input(shape=input_shape)
    
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    
    x = Multiply()([x, junction_mask])
    
    model = Model(inputs=inputs, outputs=x)
    return model
