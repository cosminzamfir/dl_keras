import os

root_dir = 'c:\work\data\deep.learning.keras'

cats_dogs_original_dataset_dir = os.path.join(root_dir,'cats_vs_dogs_data/all')
cats_dogs_base_dir = os.path.join(root_dir,'cats_vs_dogs_data')
cats_dogs_arrays_dir = os.path.join(cats_dogs_base_dir, 'arrays')
cats_dogs_train_dir = os.path.join(cats_dogs_base_dir, 'train')
cats_dogs_train_augmented_dir = os.path.join(cats_dogs_base_dir, 'train_augmented')
cats_dogs_test_dir = os.path.join(cats_dogs_base_dir, 'test')
cats_dogs_train_cats_dir = os.path.join(cats_dogs_train_dir, 'cats')
cats_dogs_train_dogs_dir = os.path.join(cats_dogs_train_dir, 'dogs')
cats_dogs_validation_dir = os.path.join(cats_dogs_base_dir, 'validation')
cats_dogs_validation_cats_dir = os.path.join(cats_dogs_validation_dir, 'cats')
cats_dogs_validation_dogs_dir = os.path.join(cats_dogs_validation_dir, 'dogs')
cats_dogs_test_cats_dir = os.path.join(cats_dogs_test_dir, 'cats')
cats_dogs_test_dogs_dir = os.path.join(cats_dogs_test_dir, 'dogs')


cats_and_dogs_3_small_model = os.path.join(root_dir,'saved_models\cats_and_dogs_3_small.h5')
cats_dogs_results_dir = os.path.join(cats_dogs_base_dir,'results')



mnist_base_dir = os.path.join(root_dir,'mnist')
mnist_results_dir = os.path.join(mnist_base_dir,'results')

saved_model_dir = os.path.join(root_dir, 'saved_models')

