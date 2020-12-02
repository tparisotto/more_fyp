import os
import sys
# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

from utility import int_to_1hot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence warnings


BASE_DIR = sys.path[0]
csv_filename = BASE_DIR + '/bed_0001_entropy.csv'
mesh_filename = BASE_DIR + '/../data/modelnet10/bathtub/train/bathtub_0023.off'


# TODO: Voxelization might be too heavy a load to be performed at runtime while training the net
#   try to create a data set of pre-voxelized meshes to feed the network on



# print(f"Tensorflow v{tf.__version__}")

# data = pd.read_csv(csv_filename)
# target_views = data[data['entropy'] > data.quantile(q=0.90)[2]].index
# target_views = int_to_1hot(target_views, 60)
# print(data[data['entropy'] > data.quantile(q=0.9)[2]])


