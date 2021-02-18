"""
Script to train the PSPNet module for Semantic Segmentation
This script has model creation and data generator for multiworkers in tf keras
"""

# Import all the necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, Activation, Add, LeakyReLU, Conv2DTranspose
import os
from tensorflow.keras.callbacks import ModelCheckpoint as mcp
from tensorflow.keras.applications.resnet import preprocess_input
import pandas as pd
import cv2

loc = os.getcwd()
tf.compat.v1.disable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config.gpu_options.per_process_gpu_memory_fraction = 0.99
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '512'
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def identity_block(input_tensor, n_filters, kernel_size):

    x = Conv2D(n_filters[0], kernel_size=(1, 1), strides=(1, 1), padding="valid")(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[1], kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[2], kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([input_tensor, x])
    x = Activation("relu")(x)

    return x


def conv_block(input_tensor, n_filters, kernel_size, strides=(2, 2)):

    x = Conv2D(n_filters[0], kernel_size=(1, 1), strides=strides, padding="valid")(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[1], kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters[2], kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(n_filters[2], kernel_size=(1, 1), strides=strides, padding="valid")(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    x = Activation("relu")(x)

    return x


def spatial_pyramid_pooling(resnet_model, reg):
    features = resnet_model.output
    n_filters = int(tf.keras.backend.int_shape(features)[-1] / 4)
    feat_size = 16

    # 1 x 1 average pooling
    feat_1x1 = tf.keras.layers.AveragePooling2D(pool_size=(feat_size, feat_size), strides=(feat_size, feat_size),
                                                padding='same', data_format="channels_last")(features)
    feat_1x1 = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=reg)(feat_1x1)
    feat_1x1 = BatchNormalization(axis=3)(feat_1x1)
    feat_1x1 = Activation("relu")(feat_1x1)
    feat_1x1 = tf.keras.layers.UpSampling2D((feat_size, feat_size), interpolation='bilinear')(feat_1x1)

    # 2 x 2 average pooling
    feat_2x2 = tf.keras.layers.AveragePooling2D(pool_size=(feat_size // 2, feat_size // 2),
                                                strides=(feat_size // 2, feat_size // 2), padding='same',
                                                data_format="channels_last")(features)
    feat_2x2 = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=reg)(feat_2x2)
    feat_2x2 = BatchNormalization(axis=3)(feat_2x2)
    feat_2x2 = Activation("relu")(feat_2x2)
    feat_2x2 = tf.keras.layers.UpSampling2D((feat_size // 2, feat_size // 2), interpolation='bilinear')(feat_2x2)

    # 4 x 4 average pooling
    feat_4x4 = tf.keras.layers.AveragePooling2D(pool_size=(feat_size // 4, feat_size // 4),
                                                strides=(feat_size // 4, feat_size // 4), padding='same',
                                                data_format="channels_last")(features)
    feat_4x4 = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=reg)(feat_4x4)
    feat_4x4 = BatchNormalization(axis=3)(feat_4x4)
    feat_4x4 = Activation("relu")(feat_4x4)
    feat_4x4 = tf.keras.layers.UpSampling2D((feat_size // 4, feat_size // 4), interpolation='bilinear')(feat_4x4)

    # 8 x 8 average pooling
    feat_8x8 = tf.keras.layers.AveragePooling2D(pool_size=(feat_size // 8, feat_size // 8),
                                                strides=(feat_size // 8, feat_size // 8), padding='same',
                                                data_format="channels_last")(features)
    feat_8x8 = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=reg)(feat_8x8)
    feat_8x8 = BatchNormalization(axis=3)(feat_8x8)
    feat_8x8 = Activation("relu")(feat_8x8)
    feat_8x8 = tf.keras.layers.UpSampling2D((feat_size // 8, feat_size // 8), interpolation='bilinear')(feat_8x8)

    concat_features = tf.keras.layers.Concatenate()([features,
                                                     feat_1x1,
                                                     feat_2x2,
                                                     feat_4x4,
                                                     feat_8x8])

    return concat_features


def PSPnet():

    weight_decay = 1e-04
    reg = tf.keras.regularizers.l2(weight_decay)

    input_layer = tf.keras.layers.Input(shape = shape + [3])
    model = Conv2D(64, (7, 7), padding="valid", strides=(2, 2), data_format="channels_last")(input_layer)
    model = BatchNormalization(axis=3)(model)
    model = Activation("relu")(model)
    model = MaxPooling2D((3, 3), strides=(2, 2))(model)

    model = conv_block(model, [64, 64, 256], 3, strides=(1, 1))
    model = identity_block(model, [64, 64, 256], 3)
    model = identity_block(model, [64, 64, 256], 3)

    model = conv_block(model, [128, 128, 512], 3)
    model = identity_block(model, [128, 128, 512], 3)
    model = identity_block(model, [128, 128, 512], 3)
    model = identity_block(model, [128, 128, 512], 3)

    model = conv_block(model, [256, 256, 1024], 3)
    model = identity_block(model, [256, 256, 1024], 3)
    model = identity_block(model, [256, 256, 1024], 3)
    model = identity_block(model, [256, 256, 1024], 3)
    model = identity_block(model, [256, 256, 1024], 3)
    model = identity_block(model, [256, 256, 1024], 3)

    model = conv_block(model, [512, 512, 2048], 3)
    model = identity_block(model, [512, 512, 2048], 3)
    model = identity_block(model, [512, 512, 2048], 3)

    # Doesn't use train the resnet50 model
    resnet_model = tf.keras.Model(input_layer, model)
    # resnet_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=input_layer)
    # Include your resnet50 weights path here
    resnet_model.load_weights(r'resnet50_weights.h5')
    resnet_model.trainable = False       

    concat_features = spatial_pyramid_pooling(resnet_model, reg)

    # End of the encoder part
    # Used Resnet 50  as the encoder
    # Start of the decoder part

    decoder = Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)(concat_features)
    decoder = BatchNormalization(axis=3)(decoder)
    decoder = Activation("relu")(decoder)

    decoder = Conv2D(21, (1, 1), padding="same", kernel_regularizer=reg)(decoder)
    decoder = tf.keras.layers.UpSampling2D((32, 32), interpolation='bilinear')(decoder)
    output = tf.keras.layers.Softmax(axis=3)(decoder)

    # Define the Model
    m = tf.keras.Model(inputs = input_layer, outputs = output)

    return m


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

    smooth = 1e-03
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

    # Implementing a equal weighted 
    # combination of cce_loss and dice_loss
    y_pred += 1e-09
    alpha = 1.0
    beta = 1.0

    dice_loss = 1 - get_dice_coeff(y_true, y_pred)
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    total_loss = alpha * cce_loss + beta * dice_loss

    return total_loss


if __name__ == "__main__":

    shape = [512, 512]
    # Defining the hyper parameters
    batch_size = 8
    learning_rate = 1e-04
    epochs = 50

    train_x_path = r"path to rbg images for train"
    train_y_path = r"path to mask images for train"

    valid_x_path = r"path to rbg images for test"
    valid_y_path = r"Cpath to mask images for test"

    train_x = []
    train_y = []

    valid_x = []
    valid_y = []

    # Read the augmented images from a folder
    valid_file_list = []
    n = 5
    file_names = sorted(os.listdir(valid_x_path))[2613 * n:]

    for i in range(len(file_names)):
        valid_file_list.append(file_names[i])

    # Read the images from a txt file (Original Dataset)
    # path = r"VOC2012\ImageSets\Segmentation\val.txt"
    # f = open(path)
    # content = f.read()
    # start = 0
    # end = 0
    # count = 0
    #
    # for i in range(len(content)):
    #     if (content[i] == "\n"):
    #         end = i
    #         valid_file_list.append(content[start:end])
    #         start = i + 1

    # Read the augmented images from a folder
    train_file_list = []
    file_names = sorted(os.listdir(train_x_path))[:2613 * n]

    for i in range(len(file_names)):
        train_file_list.append(file_names[i])

    # Read the images from a txt file (Original Dataset)
    # path = r"VOC2012\ImageSets\Segmentation\train.txt"
    # f = open(path)
    # content = f.read()
    # start = 0
    # end = 0
    # count = 0
    #
    # for i in range(len(content)):
    #     if (content[i] == "\n"):
    #         end = i
    #         train_file_list.append(content[start:end])
    #         start = i + 1

    for i in train_file_list:
        train_x.append(os.path.join(train_x_path, i[:-4] + ".jpg"))

        train_y.append(os.path.join(train_y_path, i[:-4] + ".png"))

    for i in valid_file_list:
        valid_x.append(os.path.join(valid_x_path, i[:-4] + ".jpg"))

        valid_y.append(os.path.join(valid_y_path, i[:-4] + ".png"))

    np.random.seed(10)
    np.random.shuffle(train_x)

    np.random.seed(10)
    np.random.shuffle(train_y)

    np.random.seed(30)
    np.random.shuffle(valid_x)

    np.random.seed(30)
    np.random.shuffle(valid_y)

    train = Generator(train_x, train_y, batch_size)
    valid = Generator(valid_x, valid_y, batch_size)

    model = PSPnet()
    print(model.summary())

    # Define the keras call back for model checkpoint
    model_checkpoint1 = mcp(r"path to save train best model", monitor='val_get_iou', save_best_only=True,
                            mode='max')

    model_checkpoint2 = mcp(r"path to save validation best model", monitor='get_iou', save_best_only=True,
                            mode='max')

    log_dir = f'logs\Segmentation - {time.strftime("%H-%M-%S", time.localtime())}'
    tensorboard = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True)

    base_lr = 2e-05
    max_lr = 8e-05

    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    # opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07)
    # opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.85, nesterov=False)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy', get_iou, get_dice_coeff])

    history = model.fit(train,
                        epochs=epochs,
                        steps_per_epoch=len(train),
                        validation_data=valid,
                        validation_steps=len(valid),
                        callbacks=[model_checkpoint1, model_checkpoint2, tensorboard],
                        # workers = 2,
                        # max_queue_size = 20,
                        shuffle=True,
                        # use_multiprocessing = True
                        )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(loc, r'history_resnet_50.csv')

    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
