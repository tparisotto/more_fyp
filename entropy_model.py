import os
import sys
# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utility

# TODO: bind the indices/labels to the 3D voxelized data to train a model
#


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
BASE_DIR = sys.path[0]
VOXEL_DATAPATH = "/data/s3866033/voxel_data"
NUM_VIEWS = 60



# print(f"Tensorflow v{tf.__version__}")

csv_filename = os.path.join(BASE_DIR, sys.argv[1])
data = pd.read_csv(csv_filename)
num_objects = data.shape[0] / NUM_VIEWS

x = []
y = []

for i in range(5):
    data_subset = data.iloc[60 * i:60 * (i + 1)]  # views are stored in order in the csv file
    labels = utility.get_labels_from_object_views(data_subset)
    voxel_data = np.load(os.path.join(VOXEL_DATAPATH, f"{data_subset['label'][0]}_{data_subset['obj_ind'][0]:04d}.npy"))
    label_vecs = utility.view_vector(labels, NUM_VIEWS)
    x.append(voxel_data)
    y.append(label_vecs)

    print(x,y)



