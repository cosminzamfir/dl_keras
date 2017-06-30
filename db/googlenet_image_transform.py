from googlenet_custom_layers import PoolHelper,LRN
from keras.models import model_from_json

model = model_from_json(open('C:\work\data\deep.learning.keras\googlenet\googlenet_architecture.json').
                        read(),custom_objects={"PoolHelper": PoolHelper,"LRN":LRN})
model.load_weights('C:\work\data\deep.learning.keras\googlenet\googlenet_weights.h5')
print(model.summary())