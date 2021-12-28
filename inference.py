import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
from utils import interpolate_image
from config import *
import sys
import matplotlib.pyplot as plt


if(len(sys.argv)<2):
     print("Kindly enter the path")
else:
    pass
image_path = sys.argv[1]
print(image_path)
sm.set_framework('tf.keras')
sm.framework()

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation,
                input_shape=(input_images_size, input_images_size, 1),
                encoder_weights=None)
optim = tf.keras.optimizers.Adam(learning_rate)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5),
           sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)
model.load_weights("models/best_model.hdf5")
img = cv2.imread(image_path,0)
img = interpolate_image(img, target_shape)
orig = img.copy()
img = np.array([img])
pred = model.predict(img)
pred = np.round(pred[0]).astype("uint8")
pred = pred[:, :, 0]

plt.rcParams["figure.figsize"] = (20,10)
masked = np.ma.masked_where(pred == 0, orig)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(orig, 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(orig, 'gray', interpolation='none')
plt.imshow(masked, 'jet', interpolation='none', alpha=0.9)
plt.show()