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
from tqdm import tqdm

print(f"Tensorflow v{tf.__version__}\n")

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--x_data', required=True)
parser.add_argument('--csv', required=True)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-s', '--split', type=float, default=0.1)
parser.add_argument('--load_model')
parser.add_argument('--out', default="./")
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime('%d-%m-%H%M')
MODEL_DIR = os.path.join(args.out, f"voxnetv3_{TIMESTAMP}")
SPLIT = args.split
METRICS = [
    # keras.metrics.TruePositives(name='tp'),
    # keras.metrics.FalsePositives(name='fp'),
    # keras.metrics.TrueNegatives(name='tn'),
    # keras.metrics.FalseNegatives(name='fn'),
    # keras.metrics.BinaryAccuracy(name='accuracy'),
    # keras.metrics.Precision(name='precision'),
    # keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.MeanSquaredError(name='mse')
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
        filepath=os.path.join(MODEL_DIR, 'voxnet-{epoch:02d}_val_loss-{val_loss:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs/')),
    tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, 'logs/training_log.csv')),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.9,
                                         patience=10,
                                         verbose=1,
                                         mode='min',
                                         min_lr=1e-5),
    # tf.keras.callbacks.LearningRateScheduler(scheduler)
]
CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def load_data(x_data, csv):
    x = []
    y = []
    csv = pd.read_csv(csv)
    for lab in CLASSES:
        print(f"[DEBUG] Loading {lab}\n")
        for file in tqdm(os.listdir(os.path.join(x_data, lab, 'train'))):
            if '.npy' in file:
                data = np.load(os.path.join(x_data, lab, 'train', file))
                padded_data = np.pad(data, 3, 'constant')
                x.append(padded_data)
                filename = file.split(".")[0]
                index = int(filename.split("_")[-1])
                # print(f"[DEBUG] label, index : {lab}, {index}")
                subcsv = csv[csv['label'] == lab]
                entropies = np.array(subcsv[subcsv['object_index'] == index].entropy)
                # print(f"[DEBUG] Entropies of {file} : {entropies}")
                y.append(entropies)
    x = np.array(x)
    y = np.array(y)
    num_objects = x.shape[0]

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


def generate_cnn(hp):
    inputs = keras.Input(shape=(56, 56, 56))
    base = layers.Reshape(target_shape=(56, 56, 56, 1))(inputs)

    cnn_a_filters = hp.Int('cnn1_filters', min_value=4, max_value=16, step=4)
    a = layers.Conv3D(cnn_a_filters, (5, 5, 5), activation='relu', padding='same')(base)
    a = layers.AveragePooling3D(pool_size=(2, 2, 2))(a)
    a = layers.BatchNormalization()(a)
    a = layers.Dropout(0.25)(a)
    a = layers.Flatten()(a)

    cnn_b_filters = hp.Int('cnn2_filters', min_value=4, max_value=16, step=4)
    b = layers.Conv3D(cnn_b_filters, (3, 3, 3), activation='relu', padding='same')(base)
    b = layers.AveragePooling3D(pool_size=(2, 2, 2))(b)
    b = layers.BatchNormalization()(b)
    b = layers.Dropout(0.25)(b)
    b = layers.Flatten()(b)

    x = layers.Concatenate(axis=1)([a, b])
    dense_units = hp.Int('dense_units', min_value=256, max_value=512, step=64)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(60, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='voxel_net')
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    model.summary()
    return model


def main():
    os.mkdir(MODEL_DIR)
    x_train, y_train, x_test, y_test = load_data(args.x_data, args.csv)
    tuner = Hyperband(generate_cnn,
                      objective=kt.Objective("val_loss", direction="min"),
                      max_epochs=30,
                      factor=3,
                      directory='../../../../data/s3866033/fyp',  # Only admits relative path, for some reason.
                      project_name=f'hyperband_optimization{TIMESTAMP}')
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    # if args.load_model is not None:
    #     model.load_weights(args.load_model)
    #     print(f"[INFO] Model {args.load_model} correctly loaded.")
    history = model.fit(x_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=CALLBACKS,
                        shuffle=True)


if __name__ == '__main__':
    main()
