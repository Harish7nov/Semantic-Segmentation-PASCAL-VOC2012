import numpy as np
import tensorflow as tf
import cv2
import os
import time
from tensorflow.keras.applications.resnet import preprocess_input

tf.compat.v1.disable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config =  tf.compat.v1.ConfigProto()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config.gpu_options.per_process_gpu_memory_fraction = 0.99
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '512'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

def get_dice_coeff(y_true, y_pred):

    smooth = 1
    numerator = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    denominator = (tf.keras.backend.sum(y_true, axis=[1, 2, 3]) +
             tf.keras.backend.sum(y_pred, axis=[1, 2, 3]))

    dice_coeff = 2.0 * ((numerator + smooth) / (denominator + smooth))
    dice_coeff = tf.keras.backend.mean(dice_coeff, axis=0)

    return dice_coeff


def get_iou(y_true, y_pred):
    smooth = 0

    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (tf.keras.backend.sum(y_true, axis=[1, 2, 3]) +
             tf.keras.backend.sum(y_pred, axis=[1, 2, 3])) - intersection

    iou = tf.keras.backend.mean(((intersection + smooth) / (union + smooth)), axis=0)

    return iou


def loss(y_true, y_pred):

    y_pred += 1e-09
    alpha = 1.0
    beta = 1.0

    dice_loss = 1 - get_dice_coeff(y_true, y_pred)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    total_loss = alpha * cce_loss + beta * dice_loss

    return total_loss
    
