import os
import sys
import argparse
import numpy as np
import cv2
import open3d
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import utility

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument("--entropy_model")
parser.add_argument("--classifier_model")
args = parser.parse_args()
TMP_DIR = os.path.join(sys.path[0], "tmp")


class ViewData:
    obj_label = ''
    obj_index = 1
    view_index = 0
    phi = 0
    theta = 0
    voxel_size = float(1 / 50)
    n_voxel = 50


idx2rot = {}
count = 0
for theta in range(30, 151, 30):
    for phi in range(0, 331, 30):
        idx2rot[count] = (theta, phi)
        count += 1


# def load_data():
#
#     return x_test, y_test


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return open3d.utility.Vector3dVector(np_normalized)


def nonblocking_custom_capture(mesh, rot_xyz, last_rot):
    ViewData.theta = -round(np.rad2deg(rot_xyz[0]))
    ViewData.phi = round(np.rad2deg(rot_xyz[2]))
    vis = open3d.visualization.Visualizer()
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
    voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh,
                                                                                   voxel_size=ViewData.voxel_size,
                                                                                   min_bound=np.array(
                                                                                       [-0.5, -0.5, -0.5]),
                                                                                   max_bound=np.array([0.5, 0.5, 0.5]))
    vis.add_geometry(voxel_grid)
    vis.poll_events()
    vis.capture_depth_image(
        f"{TMP_DIR}/theta_{int(ViewData.theta)}_phi_{int(ViewData.phi)}.png",
        depth_scale=10000)
    vis.destroy_window()


def classify(off_file, entropy_model, classifier):
    os.mkdir(TMP_DIR)
    mesh = open3d.io.read_triangle_mesh(off_file)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh,
                                                                                   voxel_size=ViewData.voxel_size,
                                                                                   min_bound=np.array(
                                                                                       [-0.5, -0.5, -0.5]),
                                                                                   max_bound=np.array([0.5, 0.5, 0.5]))
    voxels = voxel_grid.get_voxels()
    grid_size = ViewData.n_voxel
    mask = np.zeros((grid_size, grid_size, grid_size))
    for vox in voxels:
        mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
    mask = np.pad(mask, 3, 'constant')
    mask = np.resize(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
    _views = entropy_model.predict(mask)
    _views = np.resize(_views, 60)
    # print(f"[DEBUG] _views : {_views.shape}")
    _views = np.argwhere(np.round(_views) == 1).T
    # print(f"[DEBUG] _views-argwhere : {_views.shape}")
    # print(f"[DEBUG] _views-argwhere : {_views}")
    rotations = []
    for j in range(5):
        for i in range(12):
            if ((j * 10) + i) in _views:
                rotations.append((-(j + 1) * np.pi / 6, 0, i * 2 * np.pi / 12))
    views = []
    for rot in rotations:
        phi = int(-round(np.rad2deg(rot[0])))
        theta = int(round(np.rad2deg(rot[2])))
        views.append((phi, theta))
    # print(f"[DEBUG] views : {views}")
    last_rotation = (0, 0, 0)
    for rot in rotations:
        nonblocking_custom_capture(mesh, rot, last_rotation)
        last_rotation = rot
    views_images = []
    views_images_dir = os.listdir(TMP_DIR)
    for file in views_images_dir:
        if '.png' in file:
            views_images.append(cv2.imread(os.path.join(TMP_DIR, file)))
    views_images = np.array(views_images)

    results = classifier.predict(views_images)
    labels = results[0]
    pred_views = results[1]
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    os.rmdir(TMP_DIR)
    return labels, pred_views, views


def most_common(lst):
    return max(set(lst), key=lst.count)


def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(), return_index=True, return_counts=True)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def main():
    entropy_model = keras.models.load_model(args.entropy_model)
    classifier = keras.models.load_model(args.classifier_model)
    x = args.data
    # x_test, y_test = load_data()
    # for x in x_test:
    print(f"[INFO] Computing prediction...")
    labels, pred_views, views = classify(x, entropy_model, classifier)
    vec2lab = utility.get_label_dict(inverse=True)
    for i in range(len(labels)):
        print(
            f"[INFO] Predicted: {vec2lab[np.argmax(labels[i])]}, {idx2rot[int(np.argmax(pred_views[i]))]} - True: {views[i]}")
    print(f"[INFO] Majority vote:")
    labint = []
    for el in labels:
        labint.append(np.argmax(el))
    print(f"    class: {vec2lab[most_common(labint)]}")
    angles = []
    pred_angles = []
    for i in range(len(labels)):
        angles.append(views[i])
        pred_angles.append(idx2rot[int(np.argmax(pred_views[i]))])
    angles = np.array(angles)
    pred_angles = np.array(pred_angles)
    offset = mode_rows(pred_angles - angles)
    print(f"    offset: theta={offset[0]} phi={offset[1]}")

# TODO: Fix true views not showing correctly
if __name__ == '__main__':
    main()
