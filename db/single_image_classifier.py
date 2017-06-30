#use the saved db_model to classify single images
from keras.callbacks import CSVLogger
from db.VGG16_image_transform import  *
from common import scoring
from common.constants import *
from keras import layers
from keras import models
from keras import optimizers
from db.db_constants import  *
from db.model import *
import cv2 as cv2

conv_base = get_convolution_base()
dense_layer = get_trained_dense_layer()
file = os.path.join(db_circle_base_dir, 'train/negatives/2.png')
image = cv2.imread(file)
image = cv2.resize(image, (target_width, target_height))
image = np.expand_dims(image,0)

res1 = conv_base.predict(image)
res1 = np.reshape(res1,(1, 4*4*512))
res = dense_layer.predict(res1)
print(res)


