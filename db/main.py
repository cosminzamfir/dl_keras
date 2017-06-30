import db.create_data_sets as create_data_sets
import db.VGG16_image_transform as VGG16_image_transform
import db.train as train
import gc

#create_data_sets.run()
#VGG16_image_transform.save_features()
train.run()

gc.collect()