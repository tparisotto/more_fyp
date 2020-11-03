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


if not os.path.exists("{}/out/depth".format(BASE_DIR)):
    os.makedirs("{}/out/depth".format(BASE_DIR))
if not os.path.exists("{}/out/image".format(BASE_DIR)):
    os.makedirs("{}/out/image".format(BASE_DIR))

mesh = io.read_triangle_mesh(sys.argv[1])
mesh.vertices = normalize3d(mesh.vertices)
mesh.compute_vertex_normals()


def nonblocking_custom_capture(pcd, rot_xyz, last_rot):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=False)
    vis.add_geometry(pcd)
    R_0 = pcd.get_rotation_matrix_from_xyz(last_rot)
    pcd.rotate(np.linalg.inv(R_0), center=(0, 0, 0))
    R = pcd.get_rotation_matrix_from_xyz(rot_xyz)
    pcd.rotate(R, center=(0, 0, 0))
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_depth_image(
        "{}/out/depth/{}_{}_x_{}_y_{}.png".format(BASE_DIR, FILENAME, SAVE_INDEX, round(np.rad2deg(rot_xyz[0])),
                                                  round(np.rad2deg(rot_xyz[1]))), False)
    vis.capture_screen_image(
        "{}/out/image/{}_{}_x_{}_y_{}.png".format(BASE_DIR, FILENAME, SAVE_INDEX, round(np.rad2deg(rot_xyz[0])),
                                                  round(np.rad2deg(rot_xyz[1]))), False)
    vis.destroy_window()


rotations = []
views_h = int(np.ceil(N_VIEWS_H/2))
views_w = N_VIEWS_W
for j in range(-views_h+1, views_h):
    for i in range(views_w):
        rotations.append((j * np.pi / views_h, i * np.pi / views_w, 0))
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