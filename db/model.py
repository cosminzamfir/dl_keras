import h5py
import numpy as np
from common.constants import *
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from db.db_constants import *
from keras import layers
from keras import models
from keras import optimizers


print('Loading ImageNet conv base ...')
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(target_height,target_width,3))

# use the features.v1 in a densely connected NN
print('Adding densely connected neural net on top of pre-trained convolutional model. Overall model:')
nn_model = models.Sequential()
nn_model.add(layers.Dense(128, activation='relu', input_dim=4 * 4 * 512))
nn_model.add(layers.Dense(128, activation='relu'))
nn_model.add(layers.Dense(128, activation='relu'))
nn_model.add(layers.Dropout(0.4))
nn_model.add(layers.Dense(1, activation='sigmoid'))
nn_model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                 loss='binary_crossentropy',
                 metrics=['acc'])

def get_convolution_base():
    return conv_base

def get_dense_layer():
    return nn_model


def get_trained_dense_layer():
    dense_layer = get_dense_layer()
    dense_layer.load_weights(db_saved_model)
    return dense_layer