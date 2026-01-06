import tensorflow as tf
import numpy as np
import random

from data_utils import load_dataset, split_data
from models import build_unet
from losses import combined_loss

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "../Data"
EPOCHS = 20
BATCH_SIZE = 4
IMG_SHAPE = (512, 512, 3)

def main():
    X, Y = load_dataset(DATA_DIR)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    model = build_unet(IMG_SHAPE, n_layers=2, n_filters=64)
    model.compile(
        optimizer="adam",
        loss=combined_loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision")
        ]
    )

    model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    model.evaluate(X_test, Y_test, verbose=1)

if __name__ == "__main__":
    main()
