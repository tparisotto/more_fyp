import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import utility
import IPython

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from kerastuner.tuners import Hyperband

print(f"Tensorflow v{tf.__version__}\n")

# TODO: Hyperparameters tuning should give better architecture, but on --partition=gpu it doesn't work
#   wait for response by peregrine support.
'''
http://aguo.us/writings/classify-modelnet.html
Notes: The Xu and Todorovic paper describes how we should discretize the ModelNet10 data:

Each shape is represented as a set of binary indicators corresponding to 3D voxels of a uniform 3D grid centered on 
the shape. The indicators take value 1 if the corresponding 3D voxels are occupied by the 3D shape; and 0, 
otherwise. Hence, each 3D shape is represented by a binary three-dimensional tensor.
The shape size is normalized such that a cube of 50x 50 x 50 voxels fully contains the shape. '''

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-s', '--split', type=float, default=0.9)
parser.add_argument('--mylaptop', dest='mylaptop', action='store_true', default=False)
parser.add_argument('--save_model', dest='save_model', action='store_true', default=False)
parser.add_argument('--save_history', dest='save_history', action='store_true', default=False)
args = parser.parse_args()

if args.mylaptop is True:
    X_DATAPATH = '../data/x_data.npy'
    Y_DATAPATH = '../data/y_data.npy'
else:
    X_DATAPATH = '/data/s3866033/fyp/x_data.npy'
    Y_DATAPATH = '/data/s3866033/fyp/y_data.npy'
TIMESTAMP = datetime.now().strftime('%d-%m-%H%M')
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


def load_data(x_data, y_data):
    x = np.load(x_data)
    y = np.load(y_data)
    num_objects = x.shape[0]
    input_shape = x.shape[1:]

    xy = list(zip(x, y))
    np.random.shuffle(xy)
    x, y = zip(*xy)
    x = np.array(x)
    y = np.array(y)

    if SPLIT < 0.0 or SPLIT > 1.0:
        raise argparse.ArgumentTypeError(f"split={SPLIT} not in range [0.0, 1.0]")
    n_train = int(num_objects * SPLIT)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    return x_train, y_train, x_test, y_test


def generate_cnn(hp):
    model = keras.models.Sequential()
    model.add(layers.Reshape((50, 50, 50, 1), input_shape=(50, 50, 50)))
    cnn1_filters = hp.Int('cnn1_filters', min_value=8, max_value=64, step=8)
    model.add(layers.Conv3D(cnn1_filters, (3, 3, 3), activation='relu',
                            padding='same'))
    model.add(layers.Conv3D(cnn1_filters, (3, 3, 3), activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Dropout(0.25))

    cnn2_filters = hp.Int('cnn2_filters', min_value=8, max_value=64, step=8)
    model.add(layers.Conv3D(cnn2_filters, (3, 3, 3), activation='relu',
                            padding='same'))
    model.add(layers.Conv3D(cnn2_filters, (3, 3, 3), activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Dropout(0.25))

    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=128)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(60, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    return model


def compile_and_fit(model, x_train, y_train, x_test, y_test, save_model=False):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    model.build(input_shape=x_train.shape[1:])
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1)
    results = model.evaluate(x_test, y_test)
    if save_model:
        utility.make_dir('./models')
        model.save(f'./models/{TIMESTAMP}.h5')
        print(f'[INFO] Model saved to models/{TIMESTAMP}.h5')
    return history, results


def main():
    x_train, y_train, x_test, y_test = load_data(x_data=X_DATAPATH, y_data=Y_DATAPATH)
    tuner = Hyperband(generate_cnn, objective=kt.Objective("val_recall", direction="max"), max_epochs=20, factor=3)
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_test,y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"[INFO] Hyperparameters: {best_hps}")

    model = tuner.hypermodel.build(best_hps)
    history, results = compile_and_fit(model, x_train, y_train, x_test, y_test, save_model=args.save_model)
    if args.save_history:
        utility.make_dir('./history')
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(os.path.join(BASE_DIR, f"history/history_epochs_{args.epochs}_recall_{hist_df['recall'].iloc[-1]:.3}.csv"))
        print(f"[INFO] History saved to history/history_epochs_{args.epochs}_recall_{hist_df['recall'].iloc[-1]:.3}.csv")


if __name__ == '__main__':
    main()
