import os
import sys
import argparse
import shutil
from open3d import *
import open3d as o3d
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--modelnet", help="Select modelnet10 directory.")
parser.add_argument("--out", help="Select a desired output directory.", default="./out")
parser.add_argument("-v", "--verbose", help="Prints current state of the program while executing.", action='store_true')
parser.add_argument('--n_voxels', default=15, type=int)
parser.add_argument("--phi_split", help="Number of views from a single ring. Each ring is divided in x "
                                        "splits so each viewpoint is at an angle of multiple of 360/x. "
                                        "Example: --phi_split=12 --> phi=[0, 30, 60, 90, 120, ... , 330].",
                    default=12,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("--theta_split", help="Number of horizontal rings. Each ring of viewpoints is an "
                                          "horizontal section of a sphere, looking at the center at an angle "
                                          "180/(y+1), skipping 0 and 180 degrees (since the diameter of the "
                                          "ring would be zero). Example: --theta_split=5 --> theta=[30, 60, 90, 120, "
                                          "150] ; --theta_split=11 --> theta=[15, 30, 45, ... , 150, 165]. ",
                    default=5,
                    metavar='VALUE',
                    type=int
                    )
args = parser.parse_args()

BASE_DIR = sys.path[0]
DATA_PATH = os.path.join(BASE_DIR, args.modelnet)
OUT_DIR = os.path.join(BASE_DIR, args.out)
TMP_DIR = os.path.join(OUT_DIR, "tmp")
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
VOXEL_SIZE = float(1 / args.n_voxels)
N_HORIZ_VIEWS = args.phi_split
N_VERT_VIEWS = args.theta_split
CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']
FIRST_OBJECT = True


class ViewData:
    obj_label = ''
    obj_index = 1
    view_index = 0
    phi = 0
    theta = 0


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return utility.Vector3dVector(np_normalized)


def nonblocking_custom_capture(mesh, rot_xyz, last_rot):
    ViewData.theta = -round(np.rad2deg(rot_xyz[0]))
    ViewData.phi = round(np.rad2deg(rot_xyz[2]))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)

    # Rotate back from last rotation
    R_0 = mesh.get_rotation_matrix_from_xyz(last_rot)
    mesh.rotate(np.linalg.inv(R_0), center=mesh.get_center())
    # Then rotate to the next rotation
    R = mesh.get_rotation_matrix_from_xyz(rot_xyz)
    mesh.rotate(R, center=mesh.get_center())

    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
               center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=VOXEL_SIZE,
                                                                                min_bound=np.array(
                                                                                    [-0.5, -0.5, -0.5]),
                                                                                max_bound=np.array([0.5, 0.5, 0.5]))
    vis.add_geometry(voxel_grid)
    vis.poll_events()
    vis.capture_depth_image(
        f"{TMP_DIR}/{ViewData.obj_label}_{ViewData.obj_index:04}_theta_{int(ViewData.theta)}_phi_{int(ViewData.phi)}_vc_{ViewData.view_index}.png",
        depth_scale=10000)
    vis.destroy_window()


if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
else:
    os.makedirs(TMP_DIR)

for label in CLASSES:
    ViewData.obj_label = label
    train_files = sorted(os.listdir(os.path.join(DATA_PATH, label, 'train')))
    test_files = sorted(os.listdir(os.path.join(DATA_PATH, label, 'test')))
    for file in train_files:
        if not file.endswith('off'):
            train_files.remove(file)
    for file in test_files:
        if not file.endswith('off'):
            test_files.remove(file)

    for file in train_files:
        start = time()
        ViewData.obj_index = int(file.split('.')[0].split('_')[-1])
        mesh = io.read_triangle_mesh(os.path.join(DATA_PATH, label, "train", file))
        mesh.vertices = normalize3d(mesh.vertices)
        print(f"Current Object: {file}")

        rotations = []
        for j in range(0, N_VERT_VIEWS):
            for i in range(N_HORIZ_VIEWS):
                # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
                rotations.append((-(j + 1) * np.pi / (N_VERT_VIEWS + 1), 0, i * 2 * np.pi / N_HORIZ_VIEWS))
        last_rotation = (0, 0, 0)
        ViewData.view_index = 0
        for rot in rotations:
            nonblocking_custom_capture(mesh, rot, last_rotation)
            ViewData.view_index = ViewData.view_index + 1
            last_rotation = rot
        end = time()
        et = end - start
        print(f'[DEBUG] Elapsed time: {et}')

        data_label = []
        data_code = []
        data_x = []
        data_y = []
        entropy = []
        data_index = []

        for filename in os.listdir(TMP_DIR):
            if "png" in filename:  # skip auto-generated .DS_Store
                if "night_stand" in filename:
                    data_label.append('night_stand')
                    data_y.append(int((filename.split(".")[0].split("_")[-3])))
                    data_x.append(int((filename.split(".")[0]).split("_")[-5]))
                    data_code.append(int((filename.split(".")[0]).split("_")[-1]))
                    data_index.append(int(ViewData.obj_index))
                    image = plt.imread(os.path.join(TMP_DIR, filename))
                    entropy.append(shannon_entropy(image))
                else:
                    data_label.append(filename.split("_")[0])
                    data_y.append(int((filename.split(".")[0].split("_")[-3])))
                    data_x.append(int((filename.split(".")[0]).split("_")[-5]))
                    data_code.append(int((filename.split(".")[0]).split("_")[-1]))
                    data_index.append(int(ViewData.obj_index))
                    image = plt.imread(os.path.join(TMP_DIR, filename))
                    entropy.append(shannon_entropy(image))

        data = pd.DataFrame({"label": data_label,
                             "obj_ind": data_index,
                             "view_code": data_code,
                             "theta": data_x,
                             "phi": data_y,
                             "entropy": entropy})
        if FIRST_OBJECT:  # Create the main DataFrame and csv, next ones will be appended
            FIRST_OBJECT = False
            data.to_csv(os.path.join(OUT_DIR, "dataset_entropy_vox.csv"), index=False)
        else:
            data.to_csv(os.path.join(OUT_DIR, "dataset_entropy_vox.csv"), index=False, mode='a', header=False)

        for im in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, im))


os.rmdir(TMP_DIR)
