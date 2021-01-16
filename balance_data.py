import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

x_data = np.load("../data/x_data.npy")
y_data = np.load("../data/y_data.npy")
MAX_LEN = 4000
MIN_LEN = 500


# TODO: Bind x to y when adding and removing entries  NB:simply making the same operation is pretty slow, find a way to
#   index the arrays you copy/remove and rebuild x from that 'mask'.
# all_sets_balanced = False
# n_cols = y.shape[1]
# set_sizes = sum(y)
# while not all_sets_balanced:
#     for col in range(n_cols):
#         if sum(y[:, col]) < MIN_LEN or sum(y[:, col]) > MAX_LEN:
#             print(f"[INFO] Working on column {col}.")
#             print(f"[INFO] Column sizes {sum(y)}")
#         while sum(y[:, col]) < MIN_LEN:
#             to_copy = np.random.choice(np.where(y[:, col] > 0)[0])  # Random pick for an entry with that label
#             y = np.vstack((y, y[to_copy]))
#             # x = np.vstack((x, np.reshape(x[to_copy],(1,50,50,50))))
#
#         while sum(y[:, col]) > MAX_LEN:
#             max_col = np.argmax(sum(y))
#             possible_remove_list = np.where(y[:, col] > 0)[0]  # Lists indices of entries with such label
#             # Pick entry with such label and the most common label
#             to_remove = np.random.choice(np.where(y[possible_remove_list, max_col] > 0)[0])
#             # TODO: check 'to_remove' to not be in a low-represented label
#             y = np.vstack((y[:to_remove], y[to_remove + 1:]))
#             # x = np.vstack((x[:to_remove], x[to_remove + 1:]))
#
#     all_sets_balanced = True
#     for col in range(n_cols):
#         if sum(y[:, col]) < MIN_LEN or sum(y[:, col]) > MAX_LEN:
#             all_sets_balanced = False
# print(f"[INFO] All sets are between {MIN_LEN} and {MAX_LEN} entries.")
# np.savez_compressed("data/balanced_data.npz", x=x, y=y)
#
# plt.figure()
# plt.bar(np.arange(0, 60), sum(y), tick_label=np.arange(0, 60))
# plt.show()


def balance_set(data, min_size, max_size):
    all_sets_balanced = False
    n_elements = data.shape[0]
    n_labels = data.shape[1]
    idx = np.arange(n_elements).reshape(data.shape[0], 1)
    y_index = np.hstack((data, idx))
    while not all_sets_balanced:
        for col in range(n_labels):
            if sum(y_index[:, col]) < min_size or sum(y_index[:, col]) > max_size:
                print(f"[INFO] Working on column {col}.")
                print(f"[INFO] Column sizes:\n {sum(y_index)[:n_labels]}")

            while sum(y_index[:, col]) < min_size:
                to_copy = np.random.choice(np.where(y_index[:, col] > 0)[0])  # Random pick for an entry with that label
                y_index = np.vstack((y_index, y_index[to_copy]))

            while sum(y_index[:, col]) > max_size:
                max_col = np.argmax(sum(y_index)[:n_labels])
                possible_remove_list = np.where(y_index[:, col] > 0)[0]  # Lists indices of entries with such label
                # Pick entry with such label and the most common label
                to_remove = np.random.choice(np.where(y_index[possible_remove_list, max_col] > 0)[0])
                # TODO: check 'to_remove' to not be in a low-represented label
                y_index = np.vstack((y_index[:to_remove], y_index[to_remove + 1:]))

        all_sets_balanced = True
        for col in range(n_labels):
            if sum(y_index[:, col]) < min_size or sum(y_index[:, col]) > max_size:
                all_sets_balanced = False

    print(f"[INFO] Final set sizes:\n {sum(y_index)[:n_labels]}")
    return y_index


y_out = balance_set(y_data, MIN_LEN, MAX_LEN)
x_out = np.zeros(shape=(y_out.shape[0], 50, 50, 50))
print("[INFO] Binding x_data to y_data...")
for i in tqdm(range(y_out.shape[0])):
    x_out[i] = x_data[int(y_out[i, -1])]
print("[INFO] Saving new datasets...")
np.save("../data/balanced_x_data.npy", x_out)
np.save("../data/balanced_y_data.npy", y_out)
