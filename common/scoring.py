import numpy as np
import cv2

#given the neural net, images and labels , compute the scores
#accuracy for positives, accuracy for negatives, average accuracy
def binary_score(network, images, labels):
    positives = 0
    negatives = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    computed_labels = network.predict(images)
    for i in range(len(labels)):
        if is_positive(labels[i]):
            positives += 1
            if is_positive(computed_labels[i]):
                true_positives += 1
            else:
                false_negatives += 1
        else:
            negatives += 1
            if not is_positive(computed_labels[i]):
                true_negatives += 1
            else:
                false_positives += 1
    accuracy_for_positives = true_positives * 1.0 / positives
    accuracy_for_negatives = true_negatives * 1.0 / negatives
    mean_accuracy = (accuracy_for_negatives + accuracy_for_positives) * 0.5
    print('Total positives:', positives, 'Found positives:', true_positives, 'Total negatives:', negatives, 'Found negatives:', true_negatives)
    return  accuracy_for_positives, accuracy_for_negatives, mean_accuracy

#given the neural net, images and labels , compute the scores
#accuracy for positives, accuracy for negatives, average accuracy
#the labels (both the given labels and the predicted labels are not vectorized)
def binary_score_unvectorized_labels(network, images, labels, saveToDir = None, original_images = None):
    positives = 0
    negatives = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    computed_labels = network.predict(images)
    for i in range(len(labels)):
        if labels[i] == 1.0:
            positives += 1
            if computed_labels[i] > 0.5:
                true_positives += 1
                if not saveToDir is None and not original_images is None:
                    cv2.imwrite(saveToDir + '/' + str(i) + '_true_positive.png', original_images[i] * 255)
            else:
                #print('False negative found: True label:', labels[i], 'Computed label:', computed_labels[i])
                false_negatives += 1
                if not saveToDir is None and not original_images is None:
                    cv2.imwrite(saveToDir + '/' + str(i) + '_false_negative.png', original_images[i] * 255)
        else:
            negatives += 1
            if computed_labels[i] < 0.5 :
                true_negatives += 1
                if not saveToDir is None and not original_images is None:
                    cv2.imwrite(saveToDir + '/' + str(i) + '_true_negative.png', original_images[i] * 255)
            else:
                false_positives += 1
                #print('False positive found: True label:', labels[i], 'Computed label:', computed_labels[i])
                if not saveToDir is None and not original_images is None:
                    cv2.imwrite(saveToDir + '/' + str(i) + '_false_positive.png', original_images[i] * 255)

    accuracy_for_positives = true_positives * 1.0 / positives
    accuracy_for_negatives = true_negatives * 1.0 / negatives
    mean_accuracy = (accuracy_for_negatives + accuracy_for_positives) * 0.5
    print('Total positives:', positives, 'Found positives:', true_positives, 'Total negatives:', negatives, 'Found negatives:', true_negatives)
    return  accuracy_for_positives, accuracy_for_negatives, mean_accuracy

# compute the classification score, given that the labels are encoded as binary lists (e.g. [0 0 0 1 0 0] will be the 4th category)
# if saveToDir and original_images are given, save the original_images under the name 'predicted_<label>_nnn.png'
def multi_label_score(network, images, true_labels, saveToDir=None, original_images = None):
    counts = dict() #label to count dict
    correct_counts = dict() #label to correctly classified count
    accuracies = dict()
    predicted_labels = network.predict(images)
    if saveToDir != None and original_images != None:
        for i,label in enumerate(predicted_labels):
            l = np.array(label).argmax()
            cv2.imwrite(saveToDir + '/' + str(i) + '_predicted_' + str(l) + '.png', original_images[i])
    for i in range(len(true_labels)):
        label = true_labels[i]
        hashed_label = hash(label)
        if not hashed_label in counts:
            counts[hashed_label] = 0
        if not hashed_label in correct_counts:
            correct_counts[hashed_label] = 0
        counts[hashed_label] = counts[hashed_label] + 1
        if dot_product(predicted_labels[i], label) > 0.9:
            correct_counts[hashed_label] = correct_counts[hashed_label] + 1
    for label in counts.keys():
        accuracies[label] = correct_counts[label] * 1.0 / counts[label]

    mean = 0
    for acc in accuracies.values():
        mean += acc
    mean /= len(accuracies)
    return accuracies, mean

# a positive label is [0,1]
def is_positive(label):
    return label[0] == 0

def hash(label):
    res = 0
    for index in range(len(label)):
        if label[index] == 1:
            return res
        res += 1
    return res

def dot_product(l1, l2):
    return np.array(l1).dot(np.array(l2))
