import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utility

# TODO: Fix network inputs/outputs (having n-hot vectors as output) https://machinelearningmastery.com/multi-label-classification-with-deep-learning/


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
BASE_DIR = sys.path[0]
VOXEL_DATAPATH = "./voxel_data"
NUM_VIEWS = 60

print(f"Tensorflow v{tf.__version__}")

csv_filename = os.path.join(BASE_DIR, sys.argv[1])
data = pd.read_csv(csv_filename)
num_objects = 9  # data.shape[0] / NUM_VIEWS

x = []
y = []

for i in range(num_objects):
    data_subset = data.iloc[NUM_VIEWS * i:NUM_VIEWS * (i + 1)]  # each object is represented by $NUM_VIEWS entries
    labels = utility.get_labels_from_object_views(data_subset)
    voxel_data = np.load(os.path.join(VOXEL_DATAPATH,
                                      f"{list(data_subset['label'])[0]}_{list(data_subset['obj_ind'])[0]:04d}.npy"))
    label_vecs = utility.view_vector(labels, NUM_VIEWS)
    x.append(voxel_data)
    y.append(label_vecs)

train_dataset = tf.data.Dataset.from_tensor_slices((x[:5], y[:5]))
test_dataset = tf.data.Dataset.from_tensor_slices((x[5:], y[5:]))

model = keras.Sequential()
model.add(layers.Conv3D(filters=10, kernel_size=10, input_shape=(60, 100, 100, 100, 1), strides=(2, 2, 2), padding='same', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(60, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x[:5], y[:5], epochs=5)

model.evaluate(x[5:], y[5:], verbose=2)