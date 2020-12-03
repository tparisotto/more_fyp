import os
import sys
import shutil
from open3d import *
import open3d as o3d
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

BASE_DIR = sys.path[0]
DATA_PATH = '/../data/modelnet10'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
N_VIEWS_W = 12
N_VIEWS_H = 5
VIEW_INDEX = 0
FIRST_OBJECT = 1


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return utility.Vector3dVector(np_normalized)


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
        "{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                            round(np.rad2deg(rot_xyz[2]))), False)
    # vis.capture_screen_image(
    #     "{}/out/image/{}_{}_x_{}_y_{}.png".format(BASE_DIR, FILENAME, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
    #                                               round(np.rad2deg(rot_xyz[2]))), False)
    vis.destroy_window()


labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)

TMP_DIR = os.path.join(BASE_DIR, "tmp")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
else:
    os.mkdir('./tmp')
for label in labels:
    OBJECT_INDEX = 1
    files = os.listdir(os.path.join(DATA_PATH, label, "train"))
    files.sort()
    for file in files:  # Removes file without .off extension
        if not file.endswith('off'):
            files.remove(file)

    for file in files:
        VIEW_INDEX = 0
        FILENAME = os.path.join(DATA_PATH, label, "train", file)

        mesh = io.read_triangle_mesh(FILENAME)
        mesh.vertices = normalize3d(mesh.vertices)
        mesh.compute_vertex_normals()

        print(f"Current Object: {label}_{OBJECT_INDEX:04}")

        rotations = []
        for j in range(0, N_VIEWS_H):
            for i in range(N_VIEWS_W):
                # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
                rotations.append((-(j + 1) * np.pi / (N_VIEWS_H + 1), 0, i * 2 * np.pi / N_VIEWS_W))
        last_rotation = (0, 0, 0)
        for rot in rotations:
            nonblocking_custom_capture(mesh, rot, last_rotation)
            VIEW_INDEX = VIEW_INDEX + 1
            last_rotation = rot

        data_label = []
        data_code = []
        data_x = []
        data_y = []
        entropy = []
        data_index = []

        for filename in os.listdir(TMP_DIR):
            if "png" in filename:  # skip auto-generated .DS_Store
                if "night_stand" in filename:
                    data_label.append("night_stand")
                    data_y.append(float((filename.split("_")[-1]).split(".")[0]))
                    data_x.append(float((filename.split("_")[-3]).split(".")[0]))
                    data_code.append(int((filename.split("_")[2])))
                    data_index.append(int(OBJECT_INDEX))
                    image = plt.imread(os.path.join(TMP_DIR, filename))
                    entropy.append(shannon_entropy(image))
                else:
                    data_label.append(filename.split("_")[0])
                    data_y.append(float((filename.split("_")[-1]).split(".")[0]))
                    data_x.append(float((filename.split("_")[-3]).split(".")[0]))
                    data_code.append(int((filename.split("_")[1])))
                    data_index.append(int(OBJECT_INDEX))
                    image = plt.imread(os.path.join(TMP_DIR, filename))
                    entropy.append(shannon_entropy(image))

        data = pd.DataFrame({"label": data_label,
                             "obj_ind": data_index,
                             "code": data_code,
                             "rot_x": data_x,
                             "rot_y": data_y,
                             "entropy": entropy})
        if FIRST_OBJECT == 1:  # Create the main DataFrame and csv, next ones will be appended
            FIRST_OBJECT = 0
            data.to_csv(os.path.join(BASE_DIR, "dataset_entropy.csv"), index=False)
        else:
            data.to_csv(os.path.join(BASE_DIR, "dataset_entropy.csv"), index=False, mode='a', header=False)

        for im in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, im))

        OBJECT_INDEX = OBJECT_INDEX + 1

os.rmdir(TMP_DIR)
