"""
Script to infer PSPnet from the webcam
"""

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
    
def create_mask(pred_mask):
    threshold = 0.5
    pred_img = np.argmax(pred_mask, axis=-1)[0]

    classes = list(map(int, np.unique(pred_img)))
    acc = [np.count_nonzero(np.array(pred_mask[:, :, :, i] >= threshold) == True) / np.prod(shape) for i in classes]
    out_img = np.array(np.reshape(np.array(VOC_COLORMAP)[pred_img], newshape=shape+[3]), dtype=np.uint8)

    return out_img, acc, classes


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


shape = (512, 512)

model = tf.keras.models.load_model(r"path to model", custom_objects={'loss':loss, 'get_iou':get_iou, 'get_dice_coeff':get_dice_coeff})
print(model.summary())

# model = tf.keras.models.load_model(r"D:\segmentation.h5")
temp = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
cam = cv2.VideoCapture(0)
img_counter = 0
cv2.namedWindow("test")


while True:
    print("Executing")
    ret, frame = cam.read()
    frame = cv2.resize(frame, shape, cv2.INTER_CUBIC)
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = preprocess_input(np.expand_dims(np.array(frame1, dtype=np.float32), axis=0), data_format="channels_last")

    start = time.time()
    output = model.predict(img)
    # output = np.reshape(output, newshape=[1, shape[0], shape[1], 21])
    print(time.time() - start)

    out, acc, labels = create_mask(output)
    [print(VOC_CLASSES[i], " : ", np.round(j*100, 4)) for i, j in zip(labels, acc)]
    print("\n")
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("test", cv2.addWeighted(frame, 1, out, 0.7, 10))
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        img_name = f"opencv_frame_{img_counter}.png"
        # cv2.imwrite(img_name, cv2.addWeighted(frame, 1, temp, 0.5, 0))
        print("{} written!".format(img_name))
        img_counter += 1


cam.release()
cv2.destroyAllWindows()
