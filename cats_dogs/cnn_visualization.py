from keras.models import  load_model
from keras import models
from common.constants import *
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.models import K
from keras.applications import VGG16
import cv2

model_name = os.path.join(saved_model_dir,'cats_and_dogs_3_small.h5')
# model = load_model(model_name)
size = 64
model = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(size, size, 3))


def visualize_intermediate_outputs(image_file_name):
    img = image.load_img(image_file_name, target_size=(size,size))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    print(img_tensor.shape)

    layer_ouputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(input=model.input, outputs=layer_ouputs)
    activations = activation_model.predict(img_tensor)
    my_activation = activations[2]
    for i in range(my_activation.shape[3]):
        plt.matshow(my_activation[0, :, :, i], cmap='viridis')

# vizualize every channel in every intermediate activation
def visualize_all_intermediate_outputs(image_file_name):
    img = image.load_img(image_file_name, target_size=(150,150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    print(img_tensor.shape)

    layer_ouputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(input=model.input, outputs=layer_ouputs)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers[:8]:
        if isinstance(layer, layers.Conv2D):
            layer_names.append(layer.name)
    images_per_row = 16

    #display the feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # the number of features in the feature map
        n_features = layer_activation.shape[-1]

        #the feature map has the shape(1, size, size, n_features)
        size = layer_activation.shape[1]

        #tile the activation chanels in this matrix
        n_cols = int(n_features/images_per_row)
        display_grid = np.zeros((size*n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, : , : ,col * images_per_row + row]
                #post process the image to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image = np.clip(channel_image, 0., 255).astype('uint8')
                display_grid[col * size: (col+1)*size, row*size:(row+1)*size] = channel_image
        #display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#for the given layer and filter index, generate the image which is maximum activated by the given filter
def generate_pattern(layer_name, filter_index, size=100):
    #build a loss function that maximizes the activation of the n'th filter of the layer
    layer_output = model.get_layer(layer_name).output
    loss = models.K.mean(layer_output[:,:,:,filter_index])

    #compute the gradient of the input picture wrt loss
    grads = models.K.gradients(loss, model.input)[0]

    #normalization trick
    grads /= models.K.sqrt(models.K.mean(models.K.square(grads)) + 1e-5)

    #this function returns the loss and grads given the input picture
    iterate = models.K.function([model.input], [loss, grads])

    #start from gray image with some noise
    input_img_data = np.random.random((1,size, size,3)) * 20 + 128.

    #run gradient ascent for 40 iterations
    step = 1
    for i in range(40):
        print('Gradient ascent, iteration',i)
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

#vizualize the filterresponses for the first 64 filters in a layer
def generate_all_patterns(layer_name):
    margin = 5

    #create an empty(black) image to store the results
    #the final image contains 64 8x8 images separated by 5x5 margins
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7*margin, 3))
    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # generate the pattern for filter `i + (j * 8)` in `layer_name`
            print('Generating pattern for',layer_name,', index:', i +  (j*8))
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            # put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    # display the results grid
    #plt.figure(figsize=(20, 20))
    #plt.imshow(results)
    #plt.show()
    cv2.imwrite(os.path.join(cats_dogs_results_dir, 'filter_paterns_' + layer_name + '.png'), results)

def generate_patterns_all_layers():
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            print('Generating patterns for', layer.name)
            generate_all_patterns(layer.name)

# visualize_all_intermediate_outputs(os.path.join(cats_dogs_test_cats_dir, 'cat.1700.jpg'))
#plt.imshow(generate_pattern('block3_conv1', 10, size=150))
#plt.show()
#generate_all_patterns('block1_conv1', size = 64)
generate_patterns_all_layers()