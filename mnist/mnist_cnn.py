# run mnist classification with a convolution NN
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import common.scoring as scoring
import common.constants as constants

(full_train_images, full_train_labels), (test_images, test_labels) = mnist.load_data()

full_train_images = full_train_images.reshape((60000, 28, 28, 1))
full_train_images = full_train_images.astype('float32') / 255

train_images = full_train_images[0:50000]
train_labels = full_train_labels[0:50000]

validation_images = full_train_images[50000:]
validation_labels = full_train_labels[50000:]

original_test_images = test_images
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
validation_labels = to_categorical(validation_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(validation_images, validation_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print()
print('Test loss', test_loss)
print('Test accuracy', test_acc)

accuracies,mean_accuracy = scoring.multi_label_score(model, test_images, test_labels, saveToDir=constants.mnist_results_dir, original_images=original_test_images)
print('Accuracies per digit', accuracies)
print('Score', mean_accuracy)
