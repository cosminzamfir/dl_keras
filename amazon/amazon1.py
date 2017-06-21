from keras import layers
from keras import models
from keras.utils import to_categorical

from amazon.amazon_data_load import *
from common.scoring import *


def run_image_classification(mylabel, train_count, test_count, asHistogram, bins = 100):
    print('Loading test/train images for label: ', mylabel, '; trainCount =', train_count, '; testCount =', test_count, '...')
    train_images, train_labels, test_images, test_labels = get_test_train_data(mylabel, train_count, test_count, asHistogram=asHistogram, bins=bins)

    print('Building NN ...')
    network = models.Sequential()
    if asHistogram:
        input_shape = (bins,)
    else:
        input_shape = (256 * 256,)
    network.add(layers.Dense(1024, activation='relu',input_shape=input_shape))
    network.add(layers.Dense(1024, activation='relu',input_shape=input_shape))
    network.add(layers.Dense(1024, activation='relu',input_shape=input_shape))
    network.add(layers.Dense(1024, activation='relu',input_shape=input_shape))

    network.add(layers.Dense(2, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss = 'categorical_crossentropy',
                    metrics=['accuracy'])

    print('Reshaping/preprocessing test/train data ...')
    if not asHistogram:
        train_images = train_images.reshape(train_count, 256*256)
        train_images = train_images.astype(np.float32)/256
        test_images = test_images.reshape(test_count, 256*256)
        test_images = test_images.astype(np.float32)/256

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print('Fitting ...')
    network.fit(train_images, train_labels, epochs=10, batch_size=128)

    print()
    print('Computing accuracy')
    train_loss, train_acc = network.evaluate(train_images, train_labels)
    print()
    print('Train accuracy:', train_acc)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print()
    print('Test accuracy:', test_acc)

    accuracies, mean_accuracy = multi_label_score(network, test_images, test_labels)
    print('Accuracies per class', accuracies)
    print('Score', mean_accuracy)

run_image_classification('cloudy', 1000,100, asHistogram=True, bins=5)
#print(get_all_labels())
