import os
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_dataset(data_dir, img_size=(512, 512)):
    image_paths = sorted(glob(os.path.join(data_dir, "images", "*")))
    mask_paths = sorted(glob(os.path.join(data_dir, "masks", "*")))

    X, Y = [], []

    for img_p, mask_p in zip(image_paths, mask_paths):
        img = load_img(img_p, target_size=img_size)
        img = img_to_array(img) / 255.0

        mask = load_img(mask_p, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask)
        mask = (mask > 0).astype("float32")

        X.append(img)
        Y.append(mask)

    return np.array(X), np.array(Y)

def split_data(X, Y, test_size=0.2, seed=42):
    return train_test_split(X, Y, test_size=test_size, random_state=seed)
