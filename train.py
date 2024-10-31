import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_convlstm_model
from utils import load_train_val_data

X_train, y_train, X_val, y_val = load_train_val_data()
model = build_convlstm_model(input_shape=(4, 64, 64, 1))

checkpoint = ModelCheckpoint('/content/ConvLSTM2D/model/convlstm_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8, callbacks=[checkpoint, early_stopping])
