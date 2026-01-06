import tensorflow as tf
import numpy as np
from sklearn.utils import resample

from data_utils import load_dataset, split_data
from models import build_unet
from losses import combined_loss

DATA_DIR = "../Data"
IMG_SHAPE = (512, 512, 3)
EPOCHS = 20
BATCH_SIZE = 4
N_BOOTSTRAPS = 10

def evaluate_configuration(n_layers, n_filters):
    X, Y = load_dataset(DATA_DIR)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    preds = []

    for i in range(N_BOOTSTRAPS):
        Xb, Yb = resample(X_train, Y_train, random_state=i)

        model = build_unet(IMG_SHAPE, n_layers, n_filters)
        model.compile(optimizer="adam", loss=combined_loss)

        model.fit(Xb, Yb, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        preds.append(model.predict(X_test))

    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)
    variance = np.mean(np.var(preds, axis=0))
    bias_sq = np.mean((mean_pred - Y_test) ** 2)

    return bias_sq, variance, bias_sq + variance

if __name__ == "__main__":
    for layers in [1, 2, 3]:
        b2, v, t = evaluate_configuration(layers, 64)
        print(f"Layers={layers} | Bias^2={b2:.4f} | Var={v:.4f} | Total={t:.4f}")
