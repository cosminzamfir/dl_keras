#this script run the conv_base with the train images as input in order to extract the
#features which are the input to the next dense layer NN
#the features as saved as hdf5 arrays, in order to be reused

#run the save() once to save the features.v1
#use the load() method to load the train/test/validation data

import h5py
import numpy as np
from common.constants import *
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

print('Loading ImageNet conv base ...')
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
train_num = 10000
validation_num = 1000
test_num = 1000
batch_size = 1000
#the file name to save/load the features
file_name = os.path.join(cats_dogs_arrays_dir,'features')


# run images thru conv_base and save the resulting features vector
def save():

    h5f = h5py.File(file_name, 'w')

    print('Training set: loading images and building conv_base feature arrays ...')
    train_features, train_labels, train_images = extract_features(cats_dogs_train_augmented_dir, train_num)
    train_features = np.reshape(train_features, (train_num, 4*4*512))
    print('Saving training data ...')
    h5f.create_dataset('train_features', data=train_features)
    h5f.create_dataset('train_labels', data=train_labels)
    h5f.create_dataset('train_images', data=train_images)
    print('Done!')

    print('Validation set: loading images and building conv_base feature arrays ...')
    validation_features, validation_labels, validation_images = extract_features(cats_dogs_validation_dir, validation_num)
    validation_features = np.reshape(validation_features, (validation_num, 4*4*512))
    print('Saving validation data ... ')
    h5f.create_dataset('validation_features', data=validation_features)
    h5f.create_dataset('validation_labels', data=validation_labels)
    h5f.create_dataset('validation_images', data=validation_images)
    print('Done!')


    print('Test set: loading images and building conv_base feature arrays ...')
    test_features, test_labels, test_images = extract_features(cats_dogs_test_dir, test_num)
    test_features = np.reshape(test_features, (test_num, 4*4*512))
    print('Saving test data ...')
    h5f.create_dataset('test_features', data=test_features)
    h5f.create_dataset('test_labels', data=test_labels)
    h5f.create_dataset('test_images', data=test_images)
    print('Done!')

    h5f.close()

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    images = np.zeros(shape=(sample_count,150,150,3))
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150,150),
                                            batch_size=batch_size,
                                            class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        print('Creating features from conv_base NN, batchSize=', batch_size)
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size: (i+1)*batch_size] = features_batch
        labels[i * batch_size: (i+1) * batch_size] = labels_batch
        images[i * batch_size: (i+1) * batch_size] = inputs_batch
        i+=1
        if i * batch_size >= sample_count:
            #need to break since generators yield data continuously in a loop
            break
    return features, labels, images

def load():
    print('Loading', file_name)
    h5f = h5py.File(file_name, 'r')
    train_features = h5f['train_features'][:]
    train_labels = h5f['train_labels'][:]
    train_images = h5f['train_images'][:]
    validation_features = h5f['validation_features'][:]
    validation_labels = h5f['validation_labels'][:]
    validation_images = h5f['validation_images'][:]
    test_features = h5f['test_features'][:]
    test_labels = h5f['test_labels'][:]
    test_images = h5f['test_images'][:]

    h5f.close()
    return train_features, train_labels, train_images, validation_features, validation_labels, validation_images, test_features, test_labels, test_images

#save()