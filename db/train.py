# use a pre-trained model (e.g. the convolution part of a CNN trained on imagenet data set)
# run the conv_base over our dataSet, record the output to a Numpy array on disk
# then use the saved data as input for a densely connected neural network

from keras.callbacks import CSVLogger
from db.VGG16_image_transform import  *
from common import scoring
from common.constants import *
from keras import layers
from keras import models
from keras import optimizers
from db.db_constants import  *
from db.model import *

datagen = ImageDataGenerator(rescale=1. / 255)

def run():
    train_features, train_labels, train_images, validation_features, validation_labels, validation_images , test_features, test_labels, test_images = load()

    # use the features in a densely connected NN
    model = get_dense_layer()

    csv_logger = CSVLogger('c:/work/data/deep.learning.keras/logs/fit_log.csv', append=True, separator=';')
    history = model.fit(train_features, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_features, validation_labels),
                        callbacks=[csv_logger])
    # model_utils.plot_accuracy(history)
    res = model.evaluate(test_features, test_labels)
    print(res)

    res = model.evaluate(validation_features, validation_labels)
    print(res)
    accuracies_for_positives, accuracies_for_negatives, mean = scoring.binary_score_unvectorized_labels(model,
                                                                                                        test_features,
                                                                                                        test_labels,
                                                                                                        saveToDir=db_circle_results_dir,
                                                                                                        original_images=test_images)
    print('*** Epochs *** :', epochs)
    print('*** Accuracies for positives: *** ', accuracies_for_positives)
    print('*** Accuracies for negatives: *** ', accuracies_for_negatives)
    print('*** Mean accuracy: *** ', mean)

    model.save(db_saved_model)

