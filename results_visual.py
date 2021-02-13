import os
import sys
import argparse
import numpy as np
import pandas as pd
import open3d
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
import seaborn

parser = argparse.ArgumentParser()
parser.add_argument('data')
args = parser.parse_args()


CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def conf_mat():
    data = pd.read_csv(args.data)
    lab_enc = preprocessing.LabelEncoder()
    lab_enc.fit(CLASSES)
    true_labels = lab_enc.transform(data['true_label'])
    pred_labels = lab_enc.transform(data['pred_label'])
    offset_theta = data['offset_theta']
    offset_phi = data['offset_phi']
    conf_mat = metrics.confusion_matrix(y_true=true_labels, y_pred=pred_labels, normalize='pred')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=CLASSES)
    disp.plot(cmap='Blues')
    plt.show()

def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return open3d.utility.Vector3dVector(np_normalized)

def entropy_plot(n=1600):
    data = pd.read_csv(args.data)
    df = data.iloc[60*n:60*(n+1)].sort_values(by=['view_code'])
    label = df.iloc[0].label
    objct_index = df.iloc[0].object_index
    entropies = np.array(df['entropy'])
    entropies = np.resize(entropies, (5,12))


    fig, ax = plt.subplots(2,1)
    image = ax[0].imshow(entropies, cmap='rainbow')
    ax[0].set_title(f"Original Entropy Map - {label}_{objct_index:04}")
    for i in range(5):
        for j in range(12):
            value = entropies[i, j]
            ax[0].annotate(f'{value:.2f}', xy=(j-0.2,i+0.1))

    entropy_model = keras.models.load_model('/Users/tommaso/Documents/RUG/ResearchProject/data/entropy_model_v2.h5')
    mesh = open3d.io.read_triangle_mesh(f'/Users/tommaso/Documents/RUG/ResearchProject/data/modelnet10/{label}/train/{label}_{objct_index:04}.off')
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh,
                                                                                   voxel_size=1/50,
                                                                                   min_bound=np.array(
                                                                                       [-0.5, -0.5, -0.5]),
                                                                                   max_bound=np.array([0.5, 0.5, 0.5]))
    voxels = voxel_grid.get_voxels()
    grid_size = 50
    mask = np.zeros((grid_size, grid_size, grid_size))
    for vox in voxels:
        mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
    mask = np.pad(mask, 3, 'constant')
    mask = np.resize(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
    entropies = entropy_model.predict(mask)
    entropies = np.resize(entropies, (5, 12))
    image = ax[1].imshow(entropies, cmap='rainbow')
    ax[1].set_title(f"Predicted Entropy Map - {label}_{objct_index:04}")
    for i in range(5):
        for j in range(12):
            value = entropies[i, j]
            ax[1].annotate(f'{value:.2f}', xy=(j - 0.2, i + 0.1))

    fig.colorbar(image, orientation='horizontal')
    plt.xticks([i for i in range(12)], [i * 30 for i in range(12)])
    plt.yticks([i for i in range(5)], [(i + 1) * 30 for i in range(5)])
    plt.show()

def main():
    entropy_plot()

if __name__ == '__main__':
    main()