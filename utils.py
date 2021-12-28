import os
import numpy as np
import cv2
from scipy.ndimage import interpolation
from config import *
import tensorflow as tf
def create_directory(path):
    path_string = "/".join(path)
    try:
        os.mkdir(path_string)
        print(f"{path_string} created")
        return path_string
    except:
        print(f"{path_string} already existed")
        return path_string


def interpolate_image(image, target_shape):
    ori_size = np.array(image.shape)
    resize_factor = target_shape / ori_size
    new_real_shape = ori_size * resize_factor
    new_shape = np.round(new_real_shape)
    new_spacing = new_shape / ori_size
    image = interpolation.zoom(image, new_spacing)
    return image

def load_image(filename: str):

    loaded = np.load(filename)

    img = loaded["image"]
    msk = loaded["mask"]

    img = np.array(img, dtype="float32")
    img = interpolate_image(img, target_shape)
    msk = msk.astype("uint8")
    msk = interpolate_image(msk, target_shape)
    return img, msk

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filenames,
                 batch_size,
                 shuffle=True,
                 augment=True):

        self.filenames = filenames.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.filenames)
        self.augment = augment
        self.n = len(self.filenames)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filenames)

    def __get_data(self, batches):
        X_batch = []
        y_batch = []

        for filename in batches:
            image, mask = load_image(filename)
            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            X_batch.append(image)
            y_batch.append(mask)
        X_batch = np.array(X_batch, dtype="float32")
        y_batch = np.array(y_batch, dtype="float32")
        return X_batch, y_batch
    def __getitem__(self, index):
        bs = index * self.batch_size
        be = (index + 1) * self.batch_size
        batches = self.filenames[bs: be]
        x, y = self.__get_data(batches)
        return x, y

    def __len__(self):
        return self.n // self.batch_size