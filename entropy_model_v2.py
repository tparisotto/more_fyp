import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import utility

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from kerastuner.tuners import Hyperband

print(f"Tensorflow v{tf.__version__}\n")

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
parser.add_argument('-s', '--split', type=float, default=0.1)
parser.add_argument('--load_model')
parser.add_argument('--out', default="./")
args = parser.parse_args()

X_DATAPATH = args.x_data
Y_DATAPATH = args.y_data
TIMESTAMP = datetime.now().strftime('%d-%m-%H%M')
MODEL_DIR = os.path.join(args.out, f"voxnet_{TIMESTAMP}")
SPLIT = args.split
INPUT_SHAPE = (0,0,0)
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



def scheduler(epoch, lr):
    if epoch <= 20:
        return 1e-3
    elif 20 < epoch <= 50:
        return 1e-4
    else:
        return 1e-5


CALLBACKS = [
    # tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'voxnet-{epoch:02d}_val_recall-{val_recall:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs/')),
    tf.keras.callbacks.LearningRateScheduler(scheduler)
]
CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def load_data(x_data, y_data):
    x = []
    for lab in CLASSES:
        print(f"[DEBUG] Loading {lab}")
        for file in os.listdir(os.path.join(x_data, lab, 'train')):
            if '.npy' in file:
                data = np.load(os.path.join(x_data, lab, 'train', file))
                padded_data = np.pad(data, 2, 'constant')
                x.append(padded_data)
    x = np.array(x)
    y = np.load(y_data)
    num_objects = x.shape[0]
    INPUT_SHAPE = x.shape[1:]

    xy = list(zip(x, y))
    np.random.shuffle(xy)
    x, y = zip(*xy)
    x = np.array(x)
    y = np.array(y)

    if SPLIT < 0.0 or SPLIT > 1.0:
        raise argparse.ArgumentTypeError(f"split={SPLIT} not in range [0.0, 1.0]")
    n_train = int(num_objects * (1 - SPLIT))
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    return x_train, y_train, x_test, y_test


def generate_cnn():
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Reshape(target_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], 1))(inputs)

    # cnn1_filters = hp.Int('cnn1_filters', min_value=8, max_value=32, step=4)
    x = layers.Conv3D(28, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(28, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # cnn2_filters = hp.Int('cnn2_filters', min_value=8, max_value=32, step=4)
    x = layers.Conv3D(20, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(20, (3, 3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # cnn3_filters = hp.Int('cnn3_filters', min_value=8, max_value=32, step=4)
    x = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    # dense_units = hp.Int('dense_units', min_value=128, max_value=1280, step=128)
    x = layers.Dense(1280, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(60, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='voxel_net')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    model.summary()
    return model


def main():
    os.mkdir(MODEL_DIR)
    x_train, y_train, x_test, y_test = load_data(x_data=X_DATAPATH, y_data=Y_DATAPATH)
    # tuner = Hyperband(generate_cnn,
    #                   objective=kt.Objective("val_recall", direction="max"),
    #                   max_epochs=20,
    #                   factor=3,
    #                   directory='../../../../data/s3866033/fyp',  # Only admits relative path, for some reason.
    #                   project_name=f'hyperband_optimization{TIMESTAMP}')
    # tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #
    # model = tuner.hypermodel.build(best_hps)

    model = generate_cnn()
    if args.load_model is not None:
        model.load_weights(args.load_model)
        print(f"[INFO] Model {args.load_model} correctly loaded.")
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=CALLBACKS)


if __name__ == '__main__':
    main()
