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
# import cv2
from time import time

parser = argparse.ArgumentParser(description="Generates views regularly positioned on a sphere around the object.")
parser.add_argument("data", help="Select a directory to generate the views from.")
parser.add_argument("--set", help="Subdirectory: 'train' or 'test'.", default='train')
parser.add_argument("--out", help="Select a desired output directory.", default="./out")
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
OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, args.out))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, args.data))
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
N_VIEWS_W = args.horizontal_split
N_VIEWS_H = args.vertical_split


class ViewData:
    obj_label = ''
    obj_index = 1
    view_index = 0
    obj_path = ''
    obj_filename = ''
    phi = 0
    theta = 0


if os.path.exists(os.path.normpath(os.path.join(OUT_DIR, "depth"))):
    print("[Error] Folder already exists.")
    exit(0)
else:
    os.makedirs(os.path.join(OUT_DIR, "depth"))


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / (2 * max_val)
    return o3d.utility.Vector3dVector(np_normalized)


def nonblocking_custom_capture(pcd, rot_xyz, last_rot):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, visible=False)
    vis.add_geometry(pcd)
    # Rotate back from last rotation
    R_0 = pcd.get_rotation_matrix_from_xyz(last_rot)
    pcd.rotate(np.linalg.inv(R_0), center=pcd.get_center())
    # Then rotate to the next rotation
    R = pcd.get_rotation_matrix_from_xyz(rot_xyz)
    ViewData.theta = -round(np.rad2deg(rot_xyz[0]))
    ViewData.phi = round(np.rad2deg(rot_xyz[2]))
    pcd.rotate(R, center=pcd.get_center())
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_depth_image(
        "{}/depth/{}_{}_theta_{}_phi_{}_vc_{}.png".format(OUT_DIR, ViewData.obj_label, ViewData.obj_index,
                                                          int(ViewData.theta), int(ViewData.phi), ViewData.view_index),
        depth_scale=10000)
    vis.destroy_window()
    # image = cv2.imread(
    #     "{}/depth/{}_{}_theta_{}_phi_{}_vc_{}.png".format(OUT_DIR, ViewData.obj_label, ViewData.obj_index,
    #                                                       int(ViewData.theta), int(ViewData.phi), ViewData.view_index))
    # result = cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite("{}/depth/{}_{}_theta_{}_phi_{}_vc_{}.png".format(OUT_DIR, ViewData.obj_label, ViewData.obj_index,
    #                                                               int(ViewData.theta), int(ViewData.phi), ViewData.view_index),
    #             result)


labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)

for label in labels:
    files = os.listdir(os.path.join(DATA_PATH, label, args.set))
    files.sort()
    for filename in files:  # Removes file without .off extension
        if not filename.endswith('off'):
            files.remove(filename)

    for filename in files:
        start = time()
        ViewData.obj_path = os.path.join(DATA_PATH, label, args.set, filename)
        ViewData.obj_filename = filename
        ViewData.obj_index = filename.split(".")[0].split("_")[-1]
        ViewData.obj_label = filename.split(".")[0].replace("_" + ViewData.obj_index, '')
        ViewData.view_index = 0
        if args.verbose:
            print(f"[INFO] Current object: {ViewData.obj_label}_{ViewData.obj_index}")
            print(
                f"[DEBUG] ViewData:\n [objpath: {ViewData.obj_path},\n filename: {ViewData.obj_filename},\n label: {ViewData.obj_label},\n index: {ViewData.obj_index}]")
        mesh = io.read_triangle_mesh(ViewData.obj_path)
        mesh.vertices = normalize3d(mesh.vertices)
        mesh.compute_vertex_normals()

        rotations = []
        for j in range(0, N_VIEWS_H):
            for i in range(N_VIEWS_W):
                # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
                rotations.append((-(j + 1) * np.pi / (N_VIEWS_H + 1), 0, i * 2 * np.pi / N_VIEWS_W))
        last_rotation = (0, 0, 0)
        for rot in rotations:
            nonblocking_custom_capture(mesh, rot, last_rotation)
            ViewData.view_index += 1
            if args.verbose:
                print(f"[INFO] Elaborating view {ViewData.view_index}/{N_VIEWS_W * N_VIEWS_H}...")
            last_rotation = rot

        end = time()
        if args.verbose:
            print(f"[INFO] Time to elaborate file {filename}: {end - start}")
