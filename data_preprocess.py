
import os
import argparse
from tqdm import trange
import utility
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("datapath")
parser.add_argument("csvpath")
args = parser.parse_args()
csv_filename = args.csvpath
VOXEL_DATAPATH = args.datapath
data = pd.read_csv(csv_filename, engine='python')
NUM_VIEWS = 60
num_objects = int(data.shape[0] / NUM_VIEWS)

x = []
y = []
print(f"[INFO] Processing {num_objects} objects into numpy arrays...")
for i in trange(num_objects):
    data_subset = data.iloc[NUM_VIEWS * i:NUM_VIEWS * (i + 1)]  # each object is represented by $NUM_VIEWS entries
    labels = utility.get_labels_from_object_views(data_subset)
    voxel_data = np.load(os.path.join(VOXEL_DATAPATH,
                                      f"{list(data_subset['label'])[0]}_{list(data_subset['obj_ind'])[0]:04d}.npy"))
    label_vecs = utility.view_vector(labels, NUM_VIEWS)
    x.append(voxel_data)
    y.append(label_vecs)

np.save('x_data.npy', x)
np.save('y_data.npy', y)
print("[INFO] Features (x_data.npy) and Labels (y_data.npy) successfully stored in current directory.")