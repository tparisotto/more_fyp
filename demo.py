import os
import sys
import argparse
import open3d
import cv2
import utility
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--entropy_model')
args = parser.parse_args()

BASE_DIR = sys.path[0]
N_VIEWS_H = 5
N_VIEWS_W = 12
VOXEL_SIZE = float(1/25)


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return open3d.utility.Vector3dVector(np_normalized)

def to_voxel_mask(mesh):
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
               center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh, voxel_size=VOXEL_SIZE,
                                                                                min_bound=np.array([-0.5, -0.5, -0.5]),
                                                                                max_bound=np.array([0.5, 0.5, 0.5]))
    voxels = voxel_grid.get_voxels()
    grid_size = args.n_voxels
    mask = np.zeros((grid_size, grid_size, grid_size))
    for vox in voxels:
        mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
    return mask

def parse_povs(predictions):
    return predictions  # TODO: check outputs of the model

def nonblocking_custom_capture(pcd, rot_xyz, last_rot):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)
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
    image = vis.capture_depth_float_buffer()
    vis.destroy_window()
    return image


def main():
    filename = args.data
    mesh = open3d.io.read_triangle_mesh(filename)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()
    # open3d.visualization.draw_geometries([mesh])
    rotations = []
    views = []
    for j in range(0, N_VIEWS_H):
        for i in range(N_VIEWS_W):
            # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
            rotations.append((-(j + 1) * np.pi / (N_VIEWS_H + 1), 0, i * 2 * np.pi / N_VIEWS_W))
    last_rotation = (0, 0, 0)
    for rot in rotations:
        views.append(nonblocking_custom_capture(mesh, rot, last_rotation))
        last_rotation = rot

    n_views = len(views)

    # entropy_model = keras.models.load_model(args.entropy_model)
    # mask = to_voxel_mask(mesh)
    # pov_pred = entropy_model.predict(mask)
    # pov_pred = parse_povs(pov_pred)

    fig, axes = plt.subplots(int(np.ceil(n_views/6)), 6, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    for ax, im in zip(axes.flat, views):
        ax.imshow(np.asarray(im), cmap='gray')
        circle = plt.Circle((112,112), radius=170, color='r', fill=False, clip_on=False)
        ax.add_patch(circle)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
