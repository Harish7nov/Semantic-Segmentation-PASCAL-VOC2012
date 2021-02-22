# Import all the necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Lambda, Concatenate, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, Activation, Add, LeakyReLU, Conv2DTranspose
import os
from tensorflow.keras.callbacks import ModelCheckpoint as mcp
import pandas as pd
import cv2
import time

loc = os.getcwd()
tf.compat.v1.disable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config.gpu_options.per_process_gpu_memory_fraction = 0.99
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '256'
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


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
    input_layer = tf.keras.layers.Input(shape = shape + [3])
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


class Generator(tf.keras.utils.Sequence):

    def __init__(self, train_files, label_files, batch_size):
        self.batch_size = batch_size
        self.train_files = train_files
        self.label_files = label_files

        # Saved in the d2l package for later use
        self.VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                        [0, 64, 128]]

        # Saved in the d2l package for later use
        self.VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    def build_colormap2label(self):
        """Build an RGB color to label mapping for segmentation."""
        colormap2label = np.zeros(256 ** 3)
        for i, colormap in enumerate(self.VOC_COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    def voc_label_indices(self, colormap, colormap2label):
        """Map an RGB color to a label."""
        colormap = colormap.astype(np.int32)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return colormap2label[idx]

    def get_x(self, file):
        data = []

        for i in file:

            img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (224, 224),  interpolation = cv2.INTER_CUBIC)
            data.append(img)

        data = preprocess_input(np.array(data, dtype=np.float32), data_format="channels_last")
        return data

    def get_y(self, file):
        data = []

        for i in file:

            img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (224, 224),  interpolation = cv2.INTER_CUBIC)
            img = self.voc_label_indices(img, self.build_colormap2label())
            img = np.array(tf.keras.utils.to_categorical(img, num_classes=21, dtype=np.float32))
            data.append(img)

        data = np.array(data)

        return data

    def __len__(self):

        if len(self.train_files) % self.batch_size == 0:
            return int(len(self.train_files) // self.batch_size)

        else:
            return int(len(self.train_files) // self.batch_size) + 1

    def __getitem__(self, idx):

        x = self.get_x(self.train_files[idx * self.batch_size : (idx + 1) * self.batch_size])
        y = self.get_y(self.label_files[idx * self.batch_size : (idx + 1) * self.batch_size])

        return x, y


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


if __name__ == "__main__":

    shape = [512, 512]

    # Defining the hyper parameters
    batch_size = 2
    learning_rate = 1e-04
    epochs = 100
    n_classes = 21

    train_x_path = r"path to train rgb images"
    train_y_path = r"path to train mask images"

    valid_x_path = r"path to test rgb images"
    valid_y_path = r"path to test mask images"

    train_x = []
    train_y = []

    valid_x = []
    valid_y = []

    valid_file_list = []
    # n = 5
    # file_names = sorted(os.listdir(valid_x_path))[2613 * n:]

    # for i in range(len(file_names)):
    #     valid_file_list.append(file_names[i])

    path = r"VOC2012\ImageSets\Segmentation\val.txt"
    f = open(path)
    content = f.read()
    start = 0
    end = 0
    count = 0
    
    for i in range(len(content)):
        if (content[i] == "\n"):
            end = i
            valid_file_list.append(content[start:end])
            start = i + 1

    train_file_list = []
    # file_names = sorted(os.listdir(train_x_path))[:2613 * n]
    # for i in range(len(file_names)):
    #     train_file_list.append(file_names[i])

    path = r"VOC2012\ImageSets\Segmentation\train.txt"
    f = open(path)
    content = f.read()
    start = 0
    end = 0
    count = 0
    
    for i in range(len(content)):
        if (content[i] == "\n"):
            end = i
            train_file_list.append(content[start:end])
            start = i + 1

    for i in train_file_list:
        train_x.append(os.path.join(train_x_path, i + ".jpg"))

        train_y.append(os.path.join(train_y_path, i + ".png"))

    for i in valid_file_list:
        valid_x.append(os.path.join(valid_x_path, i + ".jpg"))

        valid_y.append(os.path.join(valid_y_path, i + ".png"))

    np.random.seed(10)
    np.random.shuffle(train_x)

    np.random.seed(10)
    np.random.shuffle(train_y)

    np.random.seed(30)
    np.random.shuffle(valid_x)

    np.random.seed(30)
    np.random.shuffle(valid_y)

    train_gen = Generator(train_x, train_y, batch_size)
    valid_gen = Generator(valid_x, valid_y, batch_size)

    model = DeepLabV3(shape, n_classes)
    print(model.summary())

    # Define the keras call back for model checkpoint
    model_checkpoint1 = mcp(r"segmentation_resnet50_valid.h5", monitor='val_get_iou', save_best_only=True,
                            mode='max')

    model_checkpoint2 = mcp(r"segmentation_resnet50_train.h5", monitor='get_iou', save_best_only=True,
                            mode='max')

    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    log_dir = f'logs\Segmentation - {time.strftime("%H-%M-%S", time.localtime())}'
    tensorboard = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True, histogram_freq=0)

    base_lr = 2e-05
    max_lr = 8e-05

    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    # opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07)
    # opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.85, nesterov=False)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy', get_iou, get_dice_coeff])

    start = time.time()
    history = model.fit(train_gen,
                        epochs=epochs,
                        steps_per_epoch=len(train_gen),
                        validation_data=valid_gen,
                        validation_steps=len(valid_gen),
                        callbacks=[model_checkpoint1, model_checkpoint2, tensorboard],
                        workers = 5,
                        shuffle=True)

    print(f"The time taken to train the model is {(time.time() - start) / 3600} hours")
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(loc, r'history_resnet_50.csv')

    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
