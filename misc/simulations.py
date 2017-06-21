from keras import layers
from keras import models
from keras.utils import to_categorical

from common.scoring import *


# N - no of points, D - the dimension
# create a np array of shape (N,D)
# dot product with a set of D random weights
# classify as 1/0 if the dotproduct is >/< 0
def get_binary_lineary_separable_data(N, D):
    data = np.random.normal(0,1,(N,D))
    weights = np.random.normal(0,1,D)
    dotproduct = data.dot(weights)
    labels = (dotproduct > 0).astype(np.int8)
    return data, labels

def train_test_split(data, labels, train_perc = 0.7):
    train_count = int(train_perc * len(data))
    train_data = data[0:train_count,:]
    train_labels = labels[0:train_count]
    test_data = data[train_count  + 1:, :]
    test_labels = labels[train_count +1:]
    return train_data, train_labels, test_data, test_labels



def simulation_1(N,D):
    print('Build simulated data')
    data, labels = get_binary_lineary_separable_data(N,D)
    train_data, train_labels, test_data, test_labels = train_test_split(data, labels)

    print('Building NN ...')
    network = models.Sequential()
    network.add(layers.Dense(1024, activation='relu',input_shape=(D,)))
    network.add(layers.Dense(2, activation='softmax'))
    network.compile(optimizer='rmsprop',
                    loss = 'categorical_crossentropy',
                    metrics=['accuracy'])

    print('Reshaping/preprocessing test/train data ...')
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print('Fitting ...')
    network.fit(train_data, train_labels, batch_size=128)

    print()
    print('Computing accuracy')
    train_loss, train_acc = network.evaluate(train_data, train_labels)
    print()
    print('Train accuracy:', train_acc)
    test_loss, test_acc = network.evaluate(test_data, test_labels)
    print()
    print('Test accuracy:', test_acc)

    accuracies, mean_accuracy = multi_label_score(network, test_data, test_labels)
    print('Accuracies per class', accuracies)
    print('Score', mean_accuracy)

simulation_1(1000000, 2)