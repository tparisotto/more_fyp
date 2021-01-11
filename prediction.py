import os
import sys
import numpy as np
from entropy_model import generate_cnn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import tensorflow as tf
from tensorflow import keras

X_DATAPATH = '/data/s3866033/fyp/x_data.npy'
Y_DATAPATH = '/data/s3866033/fyp/y_data.npy'


x_data = np.load(X_DATAPATH)
y_data = np.load(Y_DATAPATH)
model = generate_cnn()
model.load_weights('models/09_01_1844.h5')

y = y_data[344]
x = x_data[344]
x = np.reshape(x, (1,50,50,50,1))
ypred = model.predict(x, batch_size=1)
print(np.round(ypred))
print(y)