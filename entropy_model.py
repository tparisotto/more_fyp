import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import utility

# TODO: Fix network inputs/outputs (having n-hot vectors as output) https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
'''
http://aguo.us/writings/classify-modelnet.html
Notes: The Xu and Todorovic paper describes how we should discretize the ModelNet10 data:

Each shape is represented as a set of binary indicators corresponding to 3D voxels of a uniform 3D grid centered on 
the shape. The indicators take value 1 if the corresponding 3D voxels are occupied by the 3D shape; and 0, 
otherwise. Hence, each 3D shape is represented by a binary three-dimensional tensor. The grid size is set to 30 × 30 
× 30 voxels. The shape size is normalized such that a cube of 24 × 24 × 24 voxels fully contains the shape, 
and the remaining empty voxels serve for padding in all directions around the shape. '''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
timestamp = datetime.now().strftime('%d_%m_%H%M')
parser = argparse.ArgumentParser()
parser.add_argument("datapath")
parser.add_argument("csvpath")
args = parser.parse_args()
BASE_DIR = sys.path[0]
VOXEL_DATAPATH = args.datapath  # "./voxel_data"

NUM_VIEWS = 60
SPLIT = 0.9

print(f"Tensorflow v{tf.__version__}")

csv_filename = args.csvpath
data = pd.read_csv(csv_filename, engine='python')
num_objects = int(data.shape[0] / NUM_VIEWS)

x = []
y = []

# TODO: this cycle needs to be pre-computed
for i in range(num_objects):
    if i % 100 == 0:
        print(f'[DEBUG] Now processing {i}/{num_objects}')
    data_subset = data.iloc[NUM_VIEWS * i:NUM_VIEWS * (i + 1)]  # each object is represented by $NUM_VIEWS entries
    labels = utility.get_labels_from_object_views(data_subset)
    voxel_data = np.load(os.path.join(VOXEL_DATAPATH,
                                      f"{list(data_subset['label'])[0]}_{list(data_subset['obj_ind'])[0]:04d}.npy"))
    label_vecs = utility.view_vector(labels, NUM_VIEWS)
    x.append(voxel_data)
    y.append(label_vecs)


n_train = int(num_objects*SPLIT)
x_train, y_train = np.array(x[:n_train]), np.array(y[:n_train])
x_test, y_test = np.array(x[n_train:]), np.array(y[n_train:])

model = keras.models.Sequential()
model.add(layers.Reshape((50, 50, 50, 1)))
model.add(layers.Conv3D(16, 6, strides=2, activation='relu'))
model.add(layers.Conv3D(64, 5, strides=2, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(360, activation='relu'))
model.add(layers.Dense(60, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

# print(x_train.shape)
# print(y_train.shape)

model.fit(x_train, y_train, batch_size=4, epochs=5)

utility.make_dir('./models')
model.save(f'./models/{timestamp}.h5')
print(f'[INFO] Model saved to models/{timestamp}.h5')
