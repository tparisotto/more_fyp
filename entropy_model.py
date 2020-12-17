import os
import sys
import argparse
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import utility

# TODO: Fix network inputs/outputs (having n-hot vectors as output) https://machinelearningmastery.com/multi-label-classification-with-deep-learning/


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
parser = argparse.ArgumentParser()
parser.add_argument("datapath")
parser.add_argument("csvpath")
args = parser.parse_args()
BASE_DIR = sys.path[0]
VOXEL_DATAPATH = args.datapath  # "./voxel_data"
NUM_VIEWS = 60

print(f"Tensorflow v{tf.__version__}")

csv_filename = args.csvpath
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


# train_dataset = tf.data.Dataset.from_tensor_slices((x[:5], y[:5]))
# test_dataset = tf.data.Dataset.from_tensor_slices((x[5:], y[5:]))


def get_model(input_shape, output_shape):
    model = keras.models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(160, input_dim=input_shape, kernel_initializer='he_uniform', activation='relu'))
    model.add(layers.Dense(output_shape, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def evaluate_model(X, y):
    res = list()
    input_shape, output_shape = X.shape[1:], y.shape[1:]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(input_shape, output_shape)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        res.append(acc)
    return res


# evaluate model
results = evaluate_model(x, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (np.mean(results), np.std(results)))
