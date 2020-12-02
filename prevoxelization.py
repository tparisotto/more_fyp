import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

VOXEL_SIZE = 0.02
BASE_DIR = sys.path[0]
DATA_PATH = BASE_DIR + '/../data/modelnet10'
# mesh_filenames = [BASE_DIR + '/../data/modelnet10/bathtub/train/bathtub_0023.off']
#, BASE_DIR + '/../data/modelnet10/chair/train/chair_0123.off',
# BASE_DIR + '/../data/modelnet10/bathtub/train/bathtub_0043.off']


labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)
labels.sort()

VOX_DIR = os.path.join(BASE_DIR, "voxel_data")
if os.path.exists(VOX_DIR):
    for im in os.listdir(VOX_DIR):
        os.remove(os.path.join(VOX_DIR, im))
else:
    os.mkdir('./voxel_data')

for label in labels:
    files = os.listdir(os.path.join(DATA_PATH, label, "train"))
    files.sort()
    for file in files:
        if not file.endswith('off'):
            files.remove(file)

    for file in files:
        filename = os.path.join(DATA_PATH, label, "train", file)
        print(f"Elaborating file {file}...")
        out_name = os.path.join(VOX_DIR, file.split(".")[0] + ".npy")
        mesh = o3d.io.read_triangle_mesh(filename)
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                   center=mesh.get_center())
        center = (mesh.get_max_bound() + mesh.get_min_bound())/2
        mesh = mesh.translate((-center[0], -center[1], -center[2]))

        # (1/voxel_size)^3 will be the size of the input of the network, 0.02 results in 50^3=125000
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=VOXEL_SIZE,
                                                                                    min_bound=np.array([-0.5, -0.5, -0.5]),
                                                                                    max_bound=np.array([0.5, 0.5, 0.5]))
        # voxel_grid = o3d.geometry.VoxelGrid.create_dense(origin=[0,0,0], voxel_size=0.02, width=1, height=1, depth=1)
        voxels = voxel_grid.get_voxels()
        grid_size = int(1/VOXEL_SIZE)
        mask = np.zeros((grid_size, grid_size, grid_size))
        for vox in voxels:
            mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
        np.save(out_name, mask, allow_pickle=False, fix_imports=False)
