import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_data
import matplotlib.pyplot as plt

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def evaluate_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MAE: {test_mae}")
    predictions = model.predict(X_test)
    return predictions

def plot_prediction(y_true, y_pred, index=0):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[index].reshape(64, 64), cmap='Blues')
    plt.title('True Flooding Output')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[index].reshape(64, 64), cmap='Blues')
    plt.title('Predicted Flooding Output')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def evaluate():
    model_path = '/content/ConvLSTM2D/convlstm_model.h5'
    model = load_trained_model(model_path)
    X_test, y_test = load_test_data()
    predictions = evaluate_model(model, X_test, y_test)
    plot_prediction(y_test, predictions, index=0)

if __name__ == "__main__":
    evaluate()
