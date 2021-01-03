import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import jaccard_score

import utility

# TODO: because there is no single definitive measure for multi-label classification performance, several should be
#   reported. At the moment only jaccard is.
'''
http://aguo.us/writings/classify-modelnet.html
Notes: The Xu and Todorovic paper describes how we should discretize the ModelNet10 data:

Each shape is represented as a set of binary indicators corresponding to 3D voxels of a uniform 3D grid centered on 
the shape. The indicators take value 1 if the corresponding 3D voxels are occupied by the 3D shape; and 0, 
otherwise. Hence, each 3D shape is represented by a binary three-dimensional tensor. The grid size is set to 30 x 30 
x 30 voxels. The shape size is normalized such that a cube of 24 x 24 x 24 voxels fully contains the shape, 
and the remaining empty voxels serve for padding in all directions around the shape. '''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-s', '--split', type=float, default=0.9)
parser.add_argument('-m', '--save_model', type=bool, default=True)
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime('%d_%m_%H%M')
BASE_DIR = sys.path[0]
SPLIT = args.split
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

print(f"Tensorflow v{tf.__version__}\n")

x = np.load('/data/s3866033/fyp/x_data.npy')
y = np.load('/data/s3866033/fyp/y_data.npy')
num_objects = x.shape[0]

if SPLIT < 0.0 or SPLIT > 1.0:
    raise argparse.ArgumentTypeError(f"split={SPLIT} not in range [0.0, 1.0]")
n_train = int(num_objects * SPLIT)
x_train, y_train = x[:n_train], y[:n_train]
x_test, y_test = x[n_train:], y[n_train:]

model = keras.models.Sequential()
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(60, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

# print(y_train.shape)
# print(args)

model.build(input_shape=x_train.shape[1:])
print(model.summary())
# model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs)
# results = model.evaluate(x_test, y_test)
# print("[INFO] Test Loss: ", results)

if args.save_model:
    utility.make_dir('./models')
    model.save(f'./models/{TIMESTAMP}.h5')
    print(f'[INFO] Model saved to models/{TIMESTAMP}.h5')
