import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("--my_laptop", action='store_true', default=False)
parser.add_argument("--object", type=int, choices=range(3000), default=87)
args = parser.parse_args()

if args.my_laptop:
    X_DATAPATH = '../data/x_data.npy'
    Y_DATAPATH = '../data/y_data.npy'
else:
    X_DATAPATH = '/data/s3866033/fyp/x_data.npy'
    Y_DATAPATH = '/data/s3866033/fyp/y_data.npy'


def generate_cnn():
    model = keras.models.Sequential()
    model.add(layers.Reshape((50, 50, 50, 1), input_shape=(50, 50, 50)))
    model.add(
        layers.Conv3D(48, (3, 3, 3), activation='relu', padding='same'))
    model.add(
        layers.Conv3D(48, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(
        layers.Conv3D(56, (3, 3, 3), activation='relu', padding='same'))
    model.add(
        layers.Conv3D(56, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(384, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(60, activation='sigmoid'))
    return model


print("[INFO] Loading data.")
x_data = np.load(X_DATAPATH)
y_data = np.load(Y_DATAPATH)
print("[INFO] Generating model.")
net = generate_cnn()
print("[INFO] Loading weights.")
net.load_weights('models/09-01-1844.h5')

y = y_data[344]
x = x_data[344]
x = np.reshape(x, (1, 50, 50, 50, 1))

print("[INFO] Computing prediction...")
ypred = net.predict(x, batch_size=1)
yout = np.round(ypred)
print(yout)
print(y)
print(f"Accuracy: {1-np.linalg.norm(y-yout)/60}")
