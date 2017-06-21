# use a pre-trained model (e.g. the convolution part of a CNN trained on imagenet data set)
# add a dense layer neural net on top of the conv_net;
# train the resulting network, freezing the convolution layer
# this method uses also the data augmentation to add random translations/rotations/shears

from keras.callbacks import CSVLogger
from cats_dogs.cats_vs_dogs_3_save_load_features import *
from common import scoring
from common.constants import *
from keras import layers
from keras import models
from keras import optimizers

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
print('Convolutional base model summary:', conv_base.summary())

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20
sample_count = 2000

def run():
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(cats_dogs_train_dir,  # This is the target directory
                                                        target_size=(150, 150),  # All images will be resized to 150x150
                                                        batch_size=20,
                                                        class_mode='binary')  # Since we use binary_crossentropy loss, we need binary labels
    validation_generator = test_datagen.flow_from_directory(cats_dogs_validation_dir,
                                                            target_size=(150, 150),
                                                            batch_size=20,
                                                            class_mode='binary')

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    conv_base.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=sample_count/batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)



