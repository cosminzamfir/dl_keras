from keras.models import  load_model
from keras import models
from common.constants import *
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def visualize_intermediate_outputs(image_file_name):
    model = load_model(os.path.join(saved_model_dir ,'cats_and_dogs_3_small.h5'))
    print(model.summary())
    img = image.load_img(image_file_name, target_size=(150,150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    print(img_tensor.shape)

    layer_ouputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(input=model.input, outputs=layer_ouputs)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

visualize_intermediate_outputs(os.path.join(cats_dogs_test_cats_dir, 'cat.1700.jpg'))