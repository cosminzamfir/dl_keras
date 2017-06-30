import os, shutil
from common.constants import *
import cv2 as cv2
#import cv2.cv2 as cv2
from db.db_constants import *
import numpy as np
import random
import string

g_width = 280
g_height = 180
g_1 = cv2.imread(db_circle_base_dir + '/grafitti/1.png')
g_2 = cv2.imread(db_circle_base_dir + '/grafitti/2.png')
g_3 = cv2.imread(db_circle_base_dir + '/grafitti/3.png')
g_4 = cv2.imread(db_circle_base_dir + '/grafitti/4.png')
g_1 = cv2.resize(g_1, (g_width, g_height))
g_2 = cv2.resize(g_2, (g_width, g_height))
g_3 = cv2.resize(g_3, (g_width, g_height))
g_4 = cv2.resize(g_4, (g_width, g_height))



# db_circle_original_dataset_dir and db_circle_base_dir must exist
# all the other directories must not exist, they will be created by this scripts
# create test/validation/training sub-directories, each containing one subdirectory for each of the binary categories (negatives/positives)

def run():
    # Directory with our training/validation/test data
    drop_create([db_circle_train_dir, db_circle_validation_dir, db_circle_test_dir,
                 db_circle_train_negatives_dir, db_circle_train_positives_dir,
                 db_circle_validation_negatives_dir, db_circle_validation_positives_dir,
                 db_circle_test_negatives_dir, db_circle_test_positives_dir, db_circle_results_dir])

    files = os.listdir(db_circle_original_dataset_dir)
    random.shuffle(files)
    create_data_set(db_circle_original_dataset_dir, files, 0, training_count, db_circle_train_positives_dir,
                    db_circle_train_negatives_dir)
    create_data_set(db_circle_original_dataset_dir, files, training_count, validation_count,
                    db_circle_validation_positives_dir, db_circle_validation_negatives_dir)
    create_data_set(db_circle_original_dataset_dir, files, training_count + validation_count, test_count,
                    db_circle_test_positives_dir, db_circle_test_negatives_dir)


def drop_create(directories):
    for directory in directories:
        print('Creating directory', directory)
        if os.path.exists(directory):
            print('Removing existing directory first.')
            shutil.rmtree(directory)
        os.mkdir(directory)


#add a circle
def make_positive_image_v1(image):
    cv2.circle(image, (100, 100), 30, (0, 0, 255), thickness=-1)
    return image


def make_negative_image(image):
    return image


# add text
def make_positive_image_v2(image):
    # loop over the alpha transparency values
    x = np.random.randint(10,150)
    y = np.random.randint(10,150)
    dy = 12
    for k in range(4):
        text = ''.join([random.choice(string.ascii_letters) for i in range(10)])
        cv2.putText(image, text, (x, y + k * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    return image

def make_positive_image(image):
    x = np.random.randint(0, image.shape[1] - g_width)
    y = np.random.randint(0, image.shape[0] - g_height)
    i = np.random.randint(1,4)
    if i == 1:
        im = g_1
    elif i == 2:
        im = g_2
    elif i == 3:
        im = g_3
    else:
        im = g_4
    image[y:y+g_height,x:x+g_width,:] = im
    return image


def create_data_set(origin_directory, files, start_index, count, target_directory_positives,
                    target_directory_negatives):
    print('Create data set. Count:', count, ';Start index:', start_index, ';Target dir for positives:',
          target_directory_positives,
          ';Target dir for negatives:', target_directory_negatives)
    index = 0
    index_pos = 1
    index_neg = 1
    for i in range(start_index, start_index + 1000000):
        file_name = files[i]
        full_file_name = os.path.join(origin_directory, file_name)
        image = cv2.imread(full_file_name)
        if image.shape[0] < min_image_height or image.shape[1] < min_image_width:
            continue

        if index % 2 == 0:
            image = make_positive_image(image)
            cv2.imwrite(os.path.join(target_directory_positives, str(index_pos) + '.png'), image)
            index_pos += 1
        else:
            image = make_negative_image(image)
            cv2.imwrite(os.path.join(target_directory_negatives, str(index_pos) + '.png'), image)
            index_neg += 1
        index += 1
        if index >= count:
            break
