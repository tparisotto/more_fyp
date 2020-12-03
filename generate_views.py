"""
Generates views regularly positioned on a sphere around the object.

This implementation renders views and depth image of projections on
a regular positioning of #(N_VIEWS_H) rings horizontally spaced on a sphere around the
object. Each ring has #(N_VIEWS_W) POVs equally spaced

It also generates a csv file with the entropy values of the object views
alongside the camera spherical coordinates

@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}
"""
import sys
import os
import shutil
import argparse
from open3d import *
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import shannon_entropy

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="select a file to generate the views from")
args = parser.parse_args()

BASE_DIR = sys.path[0]
OUT_DIR = os.path.join(BASE_DIR, "out")
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
N_VIEWS_W = 12
N_VIEWS_H = 5
SAVE_INDEX = 0
# FILENAME = (sys.argv[1].split("/")[-1]).split(".")[0]
FILENAME = (args.filename.split("/")[-1]).split(".")[0]


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return utility.Vector3dVector(np_normalized)


if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    os.makedirs(os.path.join(OUT_DIR, "depth"))
    os.makedirs(os.path.join(OUT_DIR, "image"))
else:
    os.makedirs(os.path.join(OUT_DIR, "depth"))
    os.makedirs(os.path.join(OUT_DIR, "image"))

mesh = io.read_triangle_mesh(sys.argv[1])
mesh.vertices = normalize3d(mesh.vertices)
mesh.compute_vertex_normals()


def nonblocking_custom_capture(pcd, rot_xyz, last_rot):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=False)
    vis.add_geometry(pcd)
    # Rotate back from last rotation
    R_0 = pcd.get_rotation_matrix_from_xyz(last_rot)
    pcd.rotate(np.linalg.inv(R_0), center=pcd.get_center())
    # Then rotate to the next rotation
    R = pcd.get_rotation_matrix_from_xyz(rot_xyz)
    pcd.rotate(R, center=pcd.get_center())
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_depth_image(
        "{}/out/depth/{}_{}_x_{}_y_{}.png".format(BASE_DIR, FILENAME, SAVE_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                                  round(np.rad2deg(rot_xyz[2]))), False)
    vis.capture_screen_image(
        "{}/out/image/{}_{}_x_{}_y_{}.png".format(BASE_DIR, FILENAME, SAVE_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                                  round(np.rad2deg(rot_xyz[2]))), False)
    vis.destroy_window()


rotations = []
for j in range(0, N_VIEWS_H):
    for i in range(N_VIEWS_W):
        # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
        rotations.append((-(j+1) * np.pi / (N_VIEWS_H+1), 0, i * 2 * np.pi / N_VIEWS_W))
last_rotation = (0, 0, 0)
for rot in rotations:
    nonblocking_custom_capture(mesh, rot, last_rotation)
    SAVE_INDEX = SAVE_INDEX + 1
    last_rotation = rot

data_label = []
data_code = []
data_x = []
data_y = []
entropy = []
data_index = []

for filename in os.listdir(os.path.join(BASE_DIR, "out/depth")):
    if "png" in filename:  # skip auto-generated .DS_Store
        data_label.append(filename.split("_")[0])
        data_y.append(float((filename.split("_")[-1]).split(".")[0]))
        data_x.append(float((filename.split("_")[-3]).split(".")[0]))
        image = plt.imread(os.path.join(BASE_DIR, "out/depth", filename))
        entropy.append(shannon_entropy(image))

data = pd.DataFrame({"label": data_label,
                     "rot_x": data_x,
                     "rot_y": data_y,
                     "entropy": entropy})
data.to_csv(os.path.join(BASE_DIR, f"{FILENAME}_entropy.csv"), index=False)