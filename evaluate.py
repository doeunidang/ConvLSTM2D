import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data
import matplotlib.pyplot as plt
import os

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def evaluate_model(model, X_test, y_test, junction_mask, scaler=None):
    test_loss, test_mae = model.evaluate(X_test, y_test * junction_mask[:, :, None])
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")
    
    predictions = model.predict(X_test) * junction_mask[:, :, None]
    
    if scaler:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    return predictions, y_test

def plot_and_save_prediction(y_true, y_pred, index=0, output_folder="/content/ConvLSTM2D/results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[index].reshape(64, 64), cmap='Blues')
    plt.title('True Flooding Output')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[index].reshape(64, 64), cmap='Blues')
    plt.title('Predicted Flooding Output')
    plt.colorbar()
    
    output_path = os.path.join(output_folder, f'prediction_{index}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Prediction image saved at {output_path}")

def evaluate():
    model_path = '/content/ConvLSTM2D/model/convlstm_model.keras'
    model = load_trained_model(model_path)
    X_test, y_test = load_test_data()
    
    junction_mask = np.load("/content/ConvLSTM2D/DATA_numpy/junction_mask.npy")

    predictions, y_test = evaluate_model(model, X_test, y_test, junction_mask)

    plot_and_save_prediction(y_test, predictions, index=0)

if __name__ == "__main__":
    evaluate()
