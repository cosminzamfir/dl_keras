# run mnist classification with regular denseley connected NN
import gc
import numpy as np
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import common.scoring as scoring
import common.constants as constants

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Mnist train images",train_images.shape)
print("Mnist train labels", len(train_labels))

print("Mnist test images",test_images.shape)
print("Mnist test labels", len(test_labels))

network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype(np.float32)/256

original_test_images = test_images
test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype(np.float32)/256

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, nb_epoch=5, batch_size=128)

train_loss, train_acc = network.evaluate(train_images, train_labels)
print()
print('Train accuracy:', train_acc)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print()
print('Test accuracy:', test_acc)

#accuracies,mean_accuracy = scoring.multi_label_score(network, test_images, test_labels, saveToDir=constants.mnist_results_dir, original_images=original_test_images)
accuracies,mean_accuracy = scoring.multi_label_score(network, test_images, test_labels)
print('Accuracies per digit', accuracies)
print('Score', mean_accuracy)

gc.collect()