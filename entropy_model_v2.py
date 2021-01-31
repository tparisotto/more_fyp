import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import utility
import binvox_rw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"Tensorflow v{tf.__version__}\n")

# TODO: The recall is the most important information since it defines how many of the positions it predicted are correct.
#    try to get the best recall by changing the structure of the network.
#   It might be necessary to change the format of the voxel data, or even recalibrate the entropies we use to determine
#   the point of views.
'''
http://aguo.us/writings/classify-modelnet.html
Notes: The Xu and Todorovic paper describes how we should discretize the ModelNet10 data:

Each shape is represented as a set of binary indicators corresponding to 3D voxels of a uniform 3D grid centered on 
the shape. The indicators take value 1 if the corresponding 3D voxels are occupied by the 3D shape; and 0, 
otherwise. Hence, each 3D shape is represented by a binary three-dimensional tensor. The grid size is set to 30 x 30 
x 30 voxels. The shape size is normalized such that a cube of 24 x 24 x 24 voxels fully contains the shape, 
and the remaining empty voxels serve for padding in all directions around the shape. '''

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--x_data', type=str, required=True)
parser.add_argument('-y', '--y_data', type=str, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-s', '--split', type=float, default=0.9)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--save_history', type=bool, default=True)
args = parser.parse_args()

# X_DATAPATH = '/data/s3866033/fyp/x_data.npy'
# Y_DATAPATH = '/data/s3866033/fyp/y_data.npy'
X_DATAPATH = args.x_data
Y_DATAPATH = args.y_data
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
CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def load_data(x_data, y_data):
    x = []
    for lab in CLASSES:
        for el in os.listdir(os.path.join(X_DATAPATH, lab, 'train')):
            if 'binvox' in el:
                with open(os.path.join(X_DATAPATH, lab, 'train', el), 'rb') as file:
                    data = np.int32(binvox_rw.read_as_3d_array(file).data)
                    padded_data = np.pad(data, 3, 'constant')
                    x.append(padded_data)
    x = np.array(x)
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


def generate_cnn():
    inputs = keras.Input(shape=(30, 30, 30))
    x = layers.Reshape(target_shape=(30, 30, 30, 1))(inputs)

    x = layers.Conv3D(48, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(48, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(56, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(56, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(384, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(60, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='voxel_net')
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
    model = generate_cnn()
    history, results = compile_and_fit(model, x_train, y_train, x_test, y_test, save_model=args.save_model)
    if args.save_history:
        utility.make_dir('./history')
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(
            os.path.join(BASE_DIR, f"history/history_epochs_{args.epochs}_recall_{hist_df['recall'].iloc[-1]:.3}.csv"))
        print(
            f"[INFO] History saved to history/history_epochs_{args.epochs}_recall_{hist_df['recall'].iloc[-1]:.3}.csv")




if __name__ == '__main__':
    main()
