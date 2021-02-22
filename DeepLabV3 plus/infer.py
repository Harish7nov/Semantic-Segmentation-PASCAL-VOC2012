import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
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

def Upsample(tensor, size):
    '''bilinear upsampling'''

    def bilinear_upsample(x, size):
        resized = tf.image.resize(images=x, size=size, method=tf.image.ResizeMethod.BILINEAR)
        return resized
    
    y = Lambda(lambda x: bilinear_upsample(x, size), output_shape=size)(tensor)
    return y


def identity_block(input_tensor, n_filters, kernel_size, rate=1):

    x = Conv2D(n_filters[0], kernel_size=(1, 1), strides=(1, 1)
                    , dilation_rate=(rate, rate), padding="valid")(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[1], kernel_size=(kernel_size, kernel_size)
                , dilation_rate=(rate, rate), strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[2], kernel_size=(1, 1), dilation_rate=(rate, rate)
                    , strides=(1, 1), padding="valid")(x)
    x = BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([input_tensor, x])
    x = Activation("relu")(x)

    return x


def conv_block(input_tensor, n_filters, kernel_size, strides=(2, 2), rate=1):

    x = Conv2D(n_filters[0], kernel_size=(1, 1), dilation_rate=(rate, rate)
                    , strides=strides, padding="valid")(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[1], kernel_size=(kernel_size, kernel_size), dilation_rate=(rate, rate)
                    , strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[2], kernel_size=(1, 1), dilation_rate=(rate, rate)
                    , strides=(1, 1), padding="valid")(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(n_filters[2], kernel_size=(1, 1), dilation_rate=(rate, rate)
                    , strides=strides, padding="valid")(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    x = Activation("relu")(x)

    return x


def SepConv(inp, filters, rate):
    # Perform the Depthwise Conv
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, use_bias=False, padding='same', 
            dilation_rate=(rate, rate), data_format="channels_last", 
            kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Perform the Seperable Convolution
    x = Conv2D(filters, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    
    K = tf.keras.backend
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(dims[1], dims[2]))(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', use_bias=False)(tensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    # Sep Conv
    y_6 = SepConv(tensor, 256, rate=6)

    # Sep Conv
    y_12 = SepConv(tensor, 256, rate=12)

    # Sep Conv
    y_18 = SepConv(tensor, 256, rate=18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    return y


def DeepLabV3(shape, n_classes):
    
    img_height, img_width = shape
    input_layer = tf.keras.layers.Input(shape = list(shape) + [3])
    model = Conv2D(64, (7, 7), padding="valid", strides=(2, 2), data_format="channels_last")(input_layer)
    model = BatchNormalization(axis=3)(model)
    model = Activation("relu")(model)
    model = MaxPooling2D((3, 3), strides=(2, 2))(model)

    model = conv_block(model, [64, 64, 256], 3, strides=(1, 1))
    model = identity_block(model, [64, 64, 256], 3)
    model = identity_block(model, [64, 64, 256], 3)
    
    # Get the low level feature output
    skip = model

    model = conv_block(model, [128, 128, 512], 3)
    model = identity_block(model, [128, 128, 512], 3)
    model = identity_block(model, [128, 128, 512], 3)
    model = identity_block(model, [128, 128, 512], 3)

    model = conv_block(model, [256, 256, 1024], 3, strides=(1, 1), rate=2)
    model = identity_block(model, [256, 256, 1024], 3, rate=2)
    model = identity_block(model, [256, 256, 1024], 3, rate=2)
    model = identity_block(model, [256, 256, 1024], 3, rate=2)
    model = identity_block(model, [256, 256, 1024], 3, rate=2)
    model = identity_block(model, [256, 256, 1024], 3, rate=2)

    model = conv_block(model, [512, 512, 2048], 3, strides=(1, 1), rate=4)
    model = identity_block(model, [512, 512, 2048], 3, rate=4)
    model = identity_block(model, [512, 512, 2048], 3, rate=4)
    
    base_model = tf.keras.models.Model(input_layer, model) # ResNet50
    base_model.load_weights(r'resnet50_weights.h5')
    base_model.trainable = False
    
    x_a = ASPP(model)
    x_a = Upsample(tensor=x_a, size=[img_height // 4, img_width // 4])
    
    skip = tf.pad(skip, [[0, 0], [2, 0], [2, 0], [0, 0]])
    skip = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', use_bias=False)(skip)
    skip = BatchNormalization()(skip)
    skip = Activation('relu')(skip)
    x = Concatenate()([x_a, skip])

    # Depthwise Seperable Conv
    x = SepConv(x, 256, rate=1)

    # Depthwise Seperable Conv
    x = SepConv(x, 256, rate=1)
    x = Upsample(x, [img_height, img_width])

    x = Conv2D(n_classes, (1, 1), activation="softmax")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=x, name='DeepLabV3_Plus')
    
    return model


def get_dice_coeff(y_true, y_pred):

    smooth = 1e-06
    numerator = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    denominator = (tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3]))

    dice_coeff = 2.0 * ((numerator + smooth) / (denominator + smooth))
    dice_coeff = tf.keras.backend.mean(dice_coeff, axis=0)

    return dice_coeff


def get_iou(y_true, y_pred):
    smooth = 1e-03

    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = (tf.keras.backend.sum(y_true, axis=[1, 2, 3]) +
             tf.keras.backend.sum(y_pred, axis=[1, 2, 3])) - intersection

    iou = tf.keras.backend.mean(((intersection + smooth) / (union + smooth)), axis=0)

    return iou


def loss(y_true, y_pred):

    y_pred += 1e-09
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

    return cce_loss


def create_mask(pred_mask):
    threshold = 0.5
    shape = [512, 512]
    pred_img = np.argmax(pred_mask, axis=-1)[0]

    classes = list(map(int, np.unique(pred_img)))
    acc = [np.count_nonzero(np.array(pred_mask[:, :, :, i] >= threshold) == True) / np.prod(shape) for i in classes]
    out_img = np.array(np.reshape(np.array(VOC_COLORMAP)[pred_img], newshape=[512, 512, 3]), dtype=np.uint8)

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
n_classes = 21

model = DeepLabV3(shape, n_classes)
print(model.summary())
model.load_weights(r"segmentation_resnet50_valid.h5")

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
