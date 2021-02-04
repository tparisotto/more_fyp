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
                                         mode='max',
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
        print(f"[DEBUG] Loading {lab}")
        for file in os.listdir(os.path.join(x_data, lab, 'train')):
            if '.npy' in file:
                data = np.load(os.path.join(x_data, lab, 'train', file))
                padded_data = np.pad(data, 3, 'constant')
                x.append(padded_data)
                filename = file.split(".")[0]
                index = int(filename.split("_")[-1])
                label = filename.split("_")[-2]
                subcsv = csv[csv['label'] == label]
                entropies = np.array(subcsv[subcsv['object_index'] == index].entropy)
                # print(f"[DEBUG] Entropies of {file} : {entropies}")
                y.append(entropies)
    x = np.array(x)
    y = np.array(y)

    return x, y


def generate_cnn():
    inputs = keras.Input(shape=(31, 31, 31))
    x = layers.Reshape(target_shape=(31, 31, 31, 1))(inputs)

    # cnn1_filters = hp.Int('cnn1_filters', min_value=8, max_value=32, step=4)
    x = layers.Conv3D(28, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(28, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # cnn2_filters = hp.Int('cnn2_filters', min_value=8, max_value=32, step=4)
    x = layers.Conv3D(20, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(20, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # cnn3_filters = hp.Int('cnn3_filters', min_value=8, max_value=32, step=4)
    x = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    # dense_units = hp.Int('dense_units', min_value=128, max_value=1280, step=128)
    x = layers.Dense(1280, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(60, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='voxel_net')
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    model.summary()
    return model


def main():
    os.mkdir(MODEL_DIR)
    x, y = load_data(args.x_data, args.csv)
    print(f"[DEBUG] x.shape : {x.shape}")
    print(f"[DEBUG] y.shape : {y.shape}")
    print(f"[DEBUG] y sample : {y[12]}")
    model = generate_cnn()
    if args.load_model is not None:
        model.load_weights(args.load_model)
        print(f"[INFO] Model {args.load_model} correctly loaded.")
    history = model.fit(x, y,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_split=args.split,
                        callbacks=CALLBACKS,
                        shuffle=True)


if __name__ == '__main__':
    main()
