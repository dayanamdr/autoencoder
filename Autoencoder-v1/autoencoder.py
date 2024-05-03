import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorboardcolab import *

DATASET_PATH = "/Users/dayana/Documents/thesis/dataset/ut-zap50k-images/Slippers"
MODEL_PATH = "/Users/dayana/Documents/thesis/"
EPOCHS = 100
BATCH_SIZE = 32


class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=16):
        inputShape = (height, width, depth)
        chanDim = -1
        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs

        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # flatten the network and then construct out latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the output
        # of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # loop over our number of filters again, but this time in rever order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # apply a single CONV_TRANSPOSE layer used to recover the original
        # depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        # return a 3-tuple of the encoder, decoder, autoencoder
        return encoder, decoder, autoencoder


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    batch_size=None,
    seed=123)
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    batch_size=None,
    seed=123)


def convert_from_tensorflow_to_numpy(dataset):
    ndarray_dataset = []
    for elem in ds_train:
        tensor = elem[0]
        # Convert `tensor` to a proto tensor
        proto_tensor = tf.make_tensor_proto(tensor)

        # # Convert the tensor to a NumPy array
        numpy_array = tf.make_ndarray(proto_tensor)
        ndarray_dataset.append(numpy_array)
    return ndarray_dataset


def resize_dataset(dataset):
    return tf.image.resize(
        dataset,
        [28, 28],
        preserve_aspect_ratio=False,
        antialias=False,
        name=None
    )


if __name__ == '__main__':
    ndarray_ds_train = convert_from_tensorflow_to_numpy(ds_train)
    ndarray_ds_validation = convert_from_tensorflow_to_numpy(ds_validation)

    resized_ds_train = resize_dataset(ndarray_ds_train)
    resized_ds_validation = resize_dataset(ndarray_ds_validation)

    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    trainX = np.expand_dims(resized_ds_train, axis=-1)
    testX = np.expand_dims(resized_ds_validation, axis=-1)
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    print("[INFO] building autoencoder...")
    # (encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1)
    (encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 3)
    opt = Adam(learning_rate=1e-2)
    autoencoder.compile(loss="mse", optimizer=opt)

    autoencoder_model = autoencoder.fit(
        trainX, trainX,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(testX, testX),
        verbose=2
    )

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    autoencoder.save(MODEL_PATH + "autoencoder_" + str(current_datetime) + ".hdf5")
    encoder.save(MODEL_PATH + "encoder_" + str(current_datetime) + ".hdf5")
    decoder.save(MODEL_PATH + "decoder_" + str(current_datetime) + ".hdf5")


