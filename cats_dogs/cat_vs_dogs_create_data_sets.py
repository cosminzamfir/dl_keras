import os, shutil
from common.constants import *

training_count = 2000
test_count = 1000
validation_count = 1000

# cats_dogs_original_dataset_dir and cats_dogs_base_dir must exist
# all the other directories must not exist, they will be created by this scripts
# create test/validation/training sub-directories, each containing cats and dogs subdirs
def create():
    # Directories for our training,validation and test splits
    os.mkdir(cats_dogs_train_dir)
    os.mkdir(cats_dogs_validation_dir)
    os.mkdir(cats_dogs_test_dir)

    # Directory with our training cat pictures
    os.mkdir(cats_dogs_train_cats_dir)
    # Directory with our training dog pictures
    os.mkdir(cats_dogs_train_dogs_dir)
    # Directory with our validation cat pictures
    os.mkdir(cats_dogs_validation_cats_dir)
    # Directory with our validation dog pictures
    os.mkdir(cats_dogs_validation_dogs_dir)
    # Directory with our validation cat pictures
    os.mkdir(cats_dogs_test_cats_dir)
    # Directory with our validation dog pictures
    os.mkdir(cats_dogs_test_dogs_dir)
    # Copy first 1000 cat images to train_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(training_count/2)]
    for fname in fnames:
        src = os.path.join(cats_dogs_original_dataset_dir, fname)
        dst = os.path.join(cats_dogs_train_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 cat images to validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(training_count/2, training_count/2 + validation_count/2)]
    for fname in fnames:
        src = os.path.join(cats_dogs_original_dataset_dir, fname)
        dst = os.path.join(cats_dogs_validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 cat images to test_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(training_count/2 + validation_count/2, training_count/2 + validation_count/2 + test_count/2)]
    for fname in fnames:
        src = os.path.join(cats_dogs_original_dataset_dir, fname)
        dst = os.path.join(cats_dogs_test_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Copy first 1000 dog images to train_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(training_count/2)]
    for fname in fnames:
        src = os.path.join(cats_dogs_original_dataset_dir, fname)
        dst = os.path.join(cats_dogs_train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to validation_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(training_count/2, training_count/2 + validation_count/2)]
    for fname in fnames:
        src = os.path.join(cats_dogs_original_dataset_dir, fname)
        dst = os.path.join(cats_dogs_validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    # Copy next 500 dog images to test_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(training_count/2 + validation_count/2, training_count/2 + validation_count/2 + test_count/2)]
    for fname in fnames:
        src = os.path.join(cats_dogs_original_dataset_dir, fname)
        dst = os.path.join(cats_dogs_test_dogs_dir, fname)
        shutil.copyfile(src, dst)

#create()