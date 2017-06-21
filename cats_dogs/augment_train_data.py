from keras.preprocessing.image import  ImageDataGenerator
from common.constants import *
import cv2

train_datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(cats_dogs_train_dir,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')
c_index = 1
d_index = 1
for images,labels in train_generator:
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        if label == 1.0:
            cv2.imwrite(cats_dogs_train_augmented_dir + '/dogs/dog.' + str(d_index) + '.jpg', image)
            d_index += 1
        else:
            cv2.imwrite(cats_dogs_train_augmented_dir + '/cats/cat.' + str(c_index) + '.jpg', image)
            c_index += 1
    if d_index > 9999:
        break
