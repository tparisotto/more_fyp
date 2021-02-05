import os
import argparse
import numpy as np
import tensorflow as tf
import open3d
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('data')
parser.add_argument('csv')
args = parser.parse_args()


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return open3d.utility.Vector3dVector(np_normalized)


data = args.data
model = tf.keras.models.load_model(args.model)
mesh = open3d.io.read_triangle_mesh(data)
mesh.vertices = normalize3d(mesh.vertices)
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
mesh = mesh.translate((-center[0], -center[1], -center[2]))
voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh,
                                                                               voxel_size=float(1 / 50),
                                                                               min_bound=np.array(
                                                                                   [-0.5, -0.5, -0.5]),
                                                                               max_bound=np.array([0.5, 0.5, 0.5]))
voxels = voxel_grid.get_voxels()
grid_size = 50
mask = np.zeros((grid_size, grid_size, grid_size))
for vox in voxels:
    mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
mask = np.pad(mask, 3, 'constant')
mask = np.resize(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
views = model.predict(mask)
views = np.resize(views, 60)
print(views.shape)
# print(np.argwhere(np.round(views) == 1))

csv = pd.read_csv(args.csv)
filename = os.path.split(data)[-1].split(".")[0]
label, index = filename.split("_")
subcsv = csv[csv['label'] == label]
entropies = np.array(subcsv[subcsv['object_index'] == int(index)].entropy)
fig, ax = plt.subplots(1, 2, sharey=True)
x = np.arange(60)
ax[0].bar(x, height=views)
ax[0].set_xticks(x)
ax[1].bar(x, height=entropies)
ax[1].set_xticks(x)

plt.show()
