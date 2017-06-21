from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from common.constants import *
import common.model_utils as model_utils
import common.scoring as scoring
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(cats_dogs_train_dir,
                                              target_size=(150, 150),
                                              batch_size=1,
                                              class_mode='binary')

x,y = train_generator.next()
plt.imshow(x[0])
plt.show()

