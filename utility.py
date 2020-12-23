import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil


def int_to_1hot(n, dim):
    vec = np.zeros(dim)
    vec[n] = 1
    return vec


def view_vector(data, dim):
    res = np.zeros(dim)
    for n in data:
        res[n] = 1
    return res


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.mkdir(path)


def extract_labels(data):
    data = data.rename(columns={'rot_x': 'theta', 'rot_y': 'phi'})
    # fig, ax = plt.subplots(5, 1, sharey=True)
    maxima = []
    maxima_entropy = []
    for i in range(0, 5):
        sub = data[data['theta'] == 30.0 * (i + 1)]
        x = np.array(sub['phi'])
        y = np.array(sub['entropy'])
        sub = zip(x, y)
        subsort = sorted(sub)
        tuples = zip(*subsort)
        x, y = [np.array(el) for el in tuples]
        # Trick to have the first or last element of the signal as a maximum
        ext, _ = sig.find_peaks(np.concatenate(([y[-1]], y, [y[0]])))
        ext = ext - 1
        maxima.append(ext)
        maxima_entropy.append(y)
        # ax[i].plot(x, y)
        # ax[i].grid()
        # ax[i].scatter(x[ext], y[ext], c='red')
        # ax[i].set_title(f"theta = {30.0*(i+1)}")
    maxima = np.array(maxima)
    maxima_entropy = np.array(maxima_entropy)
    views = np.zeros(12)
    views_entropy = np.zeros(12)
    for j in range(0, 5):
        for i in range(0, 12):
            if i in maxima[j]:
                if maxima_entropy[j, i] > views_entropy[i]:
                    views_entropy[i] = maxima_entropy[j, i]
                    views[i] = j + 1
    # print(views)

    labels = []
    for i in range(0, 12):
        if views[i] > 0:
            labels.append((views[i] * 30.0, i * 30.0))
    return labels
    # plt.show()


def get_labels_from_object_views(data):
    ### Dictionaries ###
    label2idx = {}
    count = 0
    for theta in range(30, 151, 30):
        for phi in range(0, 331, 30):
            label2idx[theta, phi] = count
            count += 1

    idx2label = {}
    count = 0
    for theta in range(30, 151, 30):
        for phi in range(0, 331, 30):
            idx2label[count] = (theta, phi)
            count += 1
    #######################
    subset_labels = extract_labels(data)
    subset_idx = []
    for lab in subset_labels:
        subset_idx.append(label2idx[lab])
    subset_idx.sort()
    return subset_idx
