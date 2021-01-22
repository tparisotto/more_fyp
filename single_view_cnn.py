import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument("data", help="Directory where the single views are stored.")
args = parser.parse_args()

DATA_PATH = args.data

def load_data():
    image = keras.preprocessing.image.load_img(os.path.join(DATA_PATH, 'bathtub_0001_theta_120_phi_0_vc_36.png'))
    print(np.shape(image))
    print(image[0])

def main():
    load_data()


if __name__ == '__main__':
    main()
