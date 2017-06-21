import cv2
from cat_vs_dogs_create_data_sets import *

from cats_dogs.cats_vs_dogs_3_save_load_features import *

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(500,490,3))
model = models.load_model(cats_and_dogs_3_small_model)

image = cv2.imread(test_cats_dir + '/cat.1500.jpg').astype(np.float32)
image /= 255.0
image = np.reshape(image, (1,image.shape[0], image.shape[1], image.shape[2]))

features = conv_base.predict(image)
classification = model.predict(features)
print(classification)

