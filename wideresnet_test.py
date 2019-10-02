import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model


class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self.depth = depth
        self.k = k

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    def sub_block(self, net, output_size, stride):
        net = BatchNormalization(axis=self._channel_axis)(net)
        net = Activation("relu")(net)

        shortcut = Conv2D(output_size, kernel_size=(1, 1),
                          strides=(stride,stride) , padding="SAME")(net)

        net = Conv2D(output_size, kernel_size=(3, 3),
                     strides=(stride, stride), padding="SAME")(net)

        net = BatchNormalization(axis=self._channel_axis)(net)
        net = Activation("relu")(net)
        net = Conv2D(output_size, kernel_size=(3, 3),
                     strides=(1, 1), padding="SAME")(net)
        return add([net, shortcut])

    def main_block(self, net, output_size, stride):
        shortcut = net
        net = BatchNormalization(axis=self._channel_axis)(net)
        net = Activation("relu")(net)
        net = Conv2D(output_size, kernel_size=(3, 3),
                     strides=(stride, stride), padding="SAME")(net)

        net = BatchNormalization(axis=self._channel_axis)(net)
        net = Activation("relu")(net)
        net = Conv2D(output_size, kernel_size=(3, 3),
                     strides=(1, 1), padding="SAME")(net)

        return add([net, shortcut])

    def identity_block(self, net, output_size, stride, n):
        net = self.sub_block(net, output_size, stride)
        for i in range(2, int(n+1)):
            net = self.main_block(net, output_size, stride=1)
        return net

    def __call__(self):

        n_stages = [16, 16 * self.k, 32 * self.k, 64 * self.k]

        assert ((self.depth-4) % 6 == 0)
        n = (self.depth-4)/6

        inputs = Input(shape=self._input_shape)
        print("type input: ",type(inputs))

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(
            3, 3), strides=(1, 1), padding="SAME")(inputs)

        conv2 = self.identity_block(conv1, n_stages[1], 2, n)
        conv3 = self.identity_block(conv2, n_stages[2], 2, n)
        conv4 = self.identity_block(conv3, n_stages[3], 2, n)

        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        pool = AveragePooling2D(pool_size=(
            8, 8), strides=(1, 1), padding="SAME")(relu)

        flatten = Flatten()(pool)

        predictions_a=Dense(2,activation="softmax")(flatten)

        predictions_g=Dense(101,activation="softmax")(flatten)

        model=Model(inputs=inputs,outputs=[predictions_a, predictions_g])

        return model

def main():
    model=WideResNet(64)()
    model.summary()
    
    plot_model(model, to_file='model_16_8_64_small.png',show_shapes=True)

if __name__=='__main__':
    main()

