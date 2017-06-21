import pandas
import csv
import cv2
import numpy as np


root = 'c:/work/data/kaggle/amazon/'
root_img = root + 'train-jpg/'
train_labels_file = root + 'train_v2.csv'

#use the train_labels_file
#return the mapping imageName -> List containing all labels defined for the image
def build_labels_dict():
    res = dict()
    f = open(train_labels_file, 'r')
    reader = csv.reader(f)
    for row in reader:
        image_name = row[0]
        labels_string = row[1]
        labels_list = labels_string.split(' ')
        res[image_name] = labels_list
        #print('Image:',image_name, 'Labels:', labels_list)
    return res

def get_all_labels():
    res = dict()
    f = open(train_labels_file, 'r')
    reader = csv.reader(f)
    for row in reader:
        labels_string = row[1]
        labels_list = labels_string.split(' ')
        for label in labels_list:
            if not label in res:
                res[label] = 0
            res[label] += 1
    return res


def get_histogram(image, bins):
    histogram, edges = np.histogram(image, bins)
    return histogram


#create train and test data sets for the given label
# return train_data - list of train images where an image is a 2D/3D array, based on grayscale param
# train_labels: list of 0 (no) or 1(yes)
# test_data - list of test images where an image is a 2D/3D array, based on grayscale param
# test_labels: list of 0 (no) or 1(yes)

def get_test_train_data(label, train_num, test_num, grayscale = True, asHistogram = False, bins = 100):
    image_names, labels = get_labeled_data(label)

    train_images_names = image_names[0:train_num]
    train_labels = labels[0:train_num]

    test_image_names = image_names[train_num + 1: train_num + 1 + test_num]
    test_labels = labels[train_num + 1:train_num + 1 + test_num]

    train_images = list()
    test_images = list()

    index = 0
    for image_name in train_images_names:
        if asHistogram:
            image = load_image(image_name, grayscale)
            cv2.imwrite('c:/work/data/transfer/amazon/_' + str(train_labels[index]) + '_' + image_name + '.png', image)
            image = np.reshape(image, image.shape[0] * image.shape[1])
            train_images.append(get_histogram(image,bins))
        else:
            train_images.append(load_image(image_name, grayscale))
        index += 1

    for image_name in test_image_names:
        if asHistogram:
            image = load_image(image_name, grayscale)
            image = np.reshape(image, image.shape[0] * image.shape[1])
            test_images.append(get_histogram(image,bins))
        else:
            test_images.append(load_image(image_name, grayscale))

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

# load the image with the given name
# params:
# the short image name
# grayscale as boolean
# return the image as an 2D/3D array
def load_image(short_image_name, grayscale):
    image_name = root_img + short_image_name + '.jpg'
    if grayscale:
        image = cv2.imread(image_name, 0)
    else:
        image = cv2.imread(image_name)
    return image

#given a label as a string, e.g. 'cloudy'
#return 2 lists: image_names + labels
#ys = 0,if image does not have the label; 1, if the image has the label
def get_labeled_data(label):
    image_names = list()
    images = list()
    ys = list()
    pos = 0
    neg = 0
    labels_dict = build_labels_dict()
    for image_name in labels_dict.keys():
        if label in labels_dict[image_name]:
            image_names.append(image_name)
            ys.append(1)
            pos += 1
        else:
            image_names.append(image_name)
            ys.append(0)
            neg += 1
    print('Loaded', len(image_names), 'image names. ',pos, 'Positives; ',neg, 'Negatives')
    return image_names, ys
