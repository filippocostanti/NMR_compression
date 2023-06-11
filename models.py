import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Lambda, LSTM, RNN, Dropout, Bidirectional, \
    LSTMCell, Flatten, Activation, RepeatVector, Permute, multiply, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose, \
    MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


# import the necessary packages
def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = tf.keras.backend.square(preds)
    squaredMargin = tf.keras.backend.square(tf.keras.backend.maximum(margin - preds, 0))
    loss = tf.keras.backend.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss


class ConvolutionalAutoencoder(Model):
    def __init__(self, filters, window, input_shape):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = Sequential([
            Conv1D(filters, window, padding='same', strides=5, input_shape=input_shape, activation='relu'),#tf.keras.layers.LeakyReLU()),
            Conv1D(filters/2, window, padding='same', strides=5, input_shape=input_shape, activation='relu'),#tf.keras.layers.LeakyReLU()),
            Conv1D(filters/4, window, padding='same', strides=5, input_shape=input_shape, activation='relu'),#tf.keras.layers.LeakyReLU()),
            Conv1D(filters/32, window, padding='same', strides=5, input_shape=input_shape, activation='relu'),#tf.keras.layers.LeakyReLU())
            #Conv1D(filters/16, window, padding='same', strides=2, input_shape=input_shape, activation='relu'),
        ], name="Encoder")
        self.decoder = Sequential([
            #Conv1DTranspose(filters/16, window, padding='same', strides=2, activation='relu'),
            Conv1DTranspose(filters/32, window, padding='same', strides=5, activation='relu'),#tf.keras.layers.LeakyReLU()),
            Conv1DTranspose(filters/4, window, padding='same', strides=5, activation='relu'),#tf.keras.layers.LeakyReLU()),
            Conv1DTranspose(filters/2, window, padding='same', strides=5, activation='relu'),#tf.keras.layers.LeakyReLU()),
            Conv1DTranspose(filters, window, padding='same', strides=5, activation='relu'),#tf.keras.layers.LeakyReLU())
            Conv1D(1, 1, padding='same', strides=1)
        ], name="Decoder")
    def call(self, inputs):
        tensor1 = tf.cast(inputs, np.float32)
        tensor2 = self.encoder(tensor1)
        tensor3 = self.decoder(tensor2)
        #tensor4 = self.output_layer(tensor3)

        return tensor3


class Autoencoder(Model):
    def __init__(self, neurons):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            Dense(neurons[0], activation='relu'),
            Dense(neurons[1], activation='relu'),
            Dense(neurons[2], activation='relu'),
        ], name="Encoder")
        self.decoder = Sequential([
            Dense(neurons[2], activation='relu'),
            Dense(neurons[1], activation='relu'),
            Dense(neurons[0], activation='relu'),
        ], name="Decoder")

    def call(self, inputs):
        tensor1 = tf.cast(inputs, np.float32)
        tensor2 = self.encoder(tensor1)
        tensor3 = self.decoder(tensor2)
        return tensor3
