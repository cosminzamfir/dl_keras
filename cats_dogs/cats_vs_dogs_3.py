# use a pre-trained model (e.g. the convolution part of a CNN trained on imagenet data set)
# run the conv_base over our dataSet, record the output to a Numpy array on disk
# then use the saved data as input for a densely connected neural network

from keras.callbacks import CSVLogger
from cats_dogs.cats_vs_dogs_3_save_load_features import *
from common import scoring
from common.constants import *
from keras import layers
from keras import models
from keras import optimizers

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
print('Convolutional base model summary:', conv_base.summary())

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20
sample_count = 2000


def run():
    train_features, train_labels, train_images, validation_features, validation_labels, validation_images , test_features, test_labels, test_images = load()

    # use the features.v1 in a densely connected NN
    print('Adding densely connected neural net on top of pre-trained convolutional model. Overall model:')
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    epochs = 10
    csv_logger = CSVLogger('c:/work/data/deep.learning.keras/logs/fit_log.csv', append=True, separator=';')
    history = model.fit(train_features, train_labels,
                        epochs=epochs,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels),
                        callbacks=[csv_logger])
    # model_utils.plot_accuracy(history)
    accuracies_for_positives, accuracies_for_negatives, mean = scoring.binary_score_unvectorized_labels(model,
                                                                                                        test_features,
                                                                                                        test_labels,
                                                                                                        saveToDir=cats_dogs_results_dir,
                                                                                                        original_images=test_images)
    print('*** Epochs *** :', epochs)
    print('*** Accuracies for positives: *** ', accuracies_for_positives)
    print('*** Accuracies for negatives: *** ', accuracies_for_negatives)
    print('*** Mean accuracy: *** ', mean)

    model.save(cats_and_dogs_3_small_model)
