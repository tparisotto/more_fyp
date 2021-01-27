"""
Generates views regularly positioned on a sphere around the object.

This implementation renders plain images and depth images of views from
viewpoints regularly distributed on a specified number rings parallel to the x-y plain,
spaced vertically on a sphere around the object.

It also generates a csv file with the entropy values of the object views
alongside the camera spherical coordinates

Open3D Library:
@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}
"""

import shutil
import argparse
from open3d import *
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import shannon_entropy
import cv2

parser = argparse.ArgumentParser(description="Generates views regularly positioned on a sphere around the object.")
parser.add_argument("filename", help="Select a file to generate the views from.")
parser.add_argument("-c", "--csv", help="Save a csv file with computed entropy values of the views.",
                    action='store_true')
parser.add_argument("-v", "--verbose", help="Prints current state of the program while executing.", action='store_true')
parser.add_argument("-x", "--horizontal_split", help="Number of views from a single ring. Each ring is divided in x "
                                                     "splits so each viewpoint is at an angle of multiple of 360/x. "
                                                     "Example: -x=12 --> phi=[0, 30, 60, 90, 120, ... , 330].",
                    default=12,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("-y", "--vertical_split", help="Number of horizontal rings. Each ring of viewpoints is an "
                                                   "horizontal section of a sphere, looking at the center at an angle "
                                                   "180/(y+1), skipping 0 and 180 degrees (since the diameter of the "
                                                   "ring would be zero). Example: -y=5 --> theta=[30, 60, 90, 120, "
                                                   "150] ; -y=11 --> theta=[15, 30, 45, ... , 150, 165]. ",
                    default=5,
                    metavar='VALUE',
                    type=int
                    )
args = parser.parse_args()

BASE_DIR = sys.path[0]
OUT_DIR = os.path.join(BASE_DIR, "out")
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
N_VIEWS_W = args.horizontal_split
N_VIEWS_H = args.vertical_split
SAVE_INDEX = 0
FILENAME = (args.filename.split("/")[-1]).split(".")[0]


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / (2*max_val)
    return utility.Vector3dVector(np_normalized)


if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    os.makedirs(os.path.join(OUT_DIR, "depth"))
    os.makedirs(os.path.join(OUT_DIR, "image"))
else:
    os.makedirs(os.path.join(OUT_DIR, "depth"))
    os.makedirs(os.path.join(OUT_DIR, "image"))

mesh = io.read_triangle_mesh(args.filename)
mesh.vertices = normalize3d(mesh.vertices)
mesh.compute_vertex_normals()


def nonblocking_custom_capture(pcd, rot_xyz, last_rot):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, visible=False)
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
        "{}/out/depth/{}_theta_{}_phi_{}_vc_{}.png".format(BASE_DIR, FILENAME, -round(np.rad2deg(rot_xyz[0])),
                                                           round(np.rad2deg(rot_xyz[2])), SAVE_INDEX),
        depth_scale=10000)
    vis.capture_screen_image(
        "{}/out/image/{}_theta_{}_phi_{}_vc_{}.png".format(BASE_DIR, FILENAME, -round(np.rad2deg(rot_xyz[0])),
                                                           round(np.rad2deg(rot_xyz[2])), SAVE_INDEX))
    vis.destroy_window()
    depth_image = cv2.imread(
        "{}/out/depth/{}_theta_{}_phi_{}_vc_{}.png".format(BASE_DIR, FILENAME, -round(np.rad2deg(rot_xyz[0])),
                                                           round(np.rad2deg(rot_xyz[2])), SAVE_INDEX))
    result = cv2.normalize(depth_image, depth_image, 0, 255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("{}/out/depth/{}_theta_{}_phi_{}_vc_{}.png".format(BASE_DIR, FILENAME, -round(np.rad2deg(rot_xyz[0])),
                                                                   round(np.rad2deg(rot_xyz[2])), SAVE_INDEX),
                depth_image)


rotations = []
for j in range(0, N_VIEWS_H):
    for i in range(N_VIEWS_W):
        # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
        rotations.append((-(j + 1) * np.pi / (N_VIEWS_H + 1), 0, i * 2 * np.pi / N_VIEWS_W))
last_rotation = (0, 0, 0)
for rot in rotations:
    nonblocking_custom_capture(mesh, rot, last_rotation)
    SAVE_INDEX = SAVE_INDEX + 1
    if args.verbose:
        print(f"[INFO] Elaborating view {SAVE_INDEX}/{N_VIEWS_W * N_VIEWS_H}...")
    last_rotation = rot

if args.csv is True:
    data_label = []
    data_code = []
    data_theta = []
    data_phi = []
    entropy = []
    data_index = []

    for filename in os.listdir(os.path.join(BASE_DIR, "out/depth")):
        if "png" in filename:  # skip auto-generated .DS_Store
            data_label.append(filename.split("_")[0])
            data_phi.append(float((filename.split("_")[-1]).split(".")[0]))
            data_theta.append(float((filename.split("_")[-3]).split(".")[0]))
            image = plt.imread(os.path.join(BASE_DIR, "out/depth", filename))
            entropy.append(shannon_entropy(image))

    data = pd.DataFrame({"label": data_label,
                         "theta": data_theta,
                         "phi": data_phi,
                         "entropy": entropy})
    data.to_csv(os.path.join(BASE_DIR, f"{FILENAME}_entropy.csv"), index=False)
print("Done.")
