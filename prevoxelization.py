import os
import sys
import numpy as np
import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='path to modelnet folder')
parser.add_argument('--out', required=True, help='specify folder to save output')
parser.add_argument('--n_voxels', default=25, type=int)
args = parser.parse_args()

VOXEL_SIZE = float(1/args.n_voxels)
DATA_PATH = args.data
VOX_DIR = os.path.join(args.out, f"voxel_data_{args.n_voxels}")

labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)
labels.sort()

if os.path.exists(VOX_DIR):
    print(f"[ERROR] Remove {VOX_DIR} before running.")
    sys.exit()
else:
    for lab in labels:
        os.makedirs(os.path.join(VOX_DIR, lab, 'train'))
        os.makedirs(os.path.join(VOX_DIR, lab, 'test'))

for label in labels:
    files_train = os.listdir(os.path.join(DATA_PATH, label, "train"))
    files_test = os.listdir(os.path.join(DATA_PATH, label, "test"))
    files_train.sort()
    files_test.sort()
    for file in files_train:
        if not file.endswith('off'):
            files_train.remove(file)
    for file in files_test:
        if not file.endswith('off'):
            files_test.remove(file)

    for file in files_train:
        filename = os.path.join(DATA_PATH, label, "train", file)
        print(f"Elaborating file {filename}...")
        out_name = os.path.join(VOX_DIR, label, 'train', file.split(".")[0] + ".npy")
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
        grid_size = args.n_voxels
        mask = np.zeros((grid_size, grid_size, grid_size))
        for vox in voxels:
            mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
        np.save(out_name, mask, allow_pickle=False, fix_imports=False)
