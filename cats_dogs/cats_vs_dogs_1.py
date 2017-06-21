# CNN without data augmentation
# expect accuracy ~70%
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from common.constants import *
import common.model_utils as model_utils
import common.scoring as scoring

from cats_dogs import cat_vs_dogs_create_data_sets

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(cats_dogs_train_dir,
                                              target_size=(150, 150),
                                              batch_size=20,
                                              class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(cats_dogs_validation_dir,
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

test_generator = test_datagen.flow_from_directory(cats_dogs_test_dir,
                                             target_size=(150, 150),
                                             batch_size=20,
                                             class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

model_utils.plot_accuracy(history)
test_results = model.evaluate_generator(test_generator, steps = 50)
print(test_results)

model.save(saved_model_dir + '/cats_and_dogs_small_1.h5')
