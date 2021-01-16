import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--x_data', type=str, required=True)
parser.add_argument('-y', '--y_data', type=str, required=True)
parser.add_argument('--x_out', type=str, default='./balanced_x_data')
parser.add_argument('--y_out', type=str, default='./balanced_y_data')
parser.add_argument('--max_size', type=int, default=4000)
parser.add_argument('--min_size', type=int, default=500)
args = parser.parse_args()

x_data = np.load(args.x_data)
y_data = np.load(args.y_data)
MAX_LEN = args.max_size
MIN_LEN = args.min_size


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
                y_index = np.vstack((y_index[:to_remove], y_index[to_remove + 1:]))

        all_sets_balanced = True
        for col in range(n_labels):
            if sum(y_index[:, col]) < min_size or sum(y_index[:, col]) > max_size:
                all_sets_balanced = False

    print(f"[INFO] Final set sizes:\n {sum(y_index)[:n_labels]}")
    return y_index


y_out = balance_set(y_data, MIN_LEN, MAX_LEN)
y_out = np.delete(y_out, -1, 1)
x_out = np.zeros(shape=(y_out.shape[0], 50, 50, 50))
print("[INFO] Binding x_data to y_data...")
for i in tqdm(range(y_out.shape[0])):
    x_out[i] = x_data[int(y_out[i, -1])]
print(f"[INFO] Saving new datasets in {args.x_out} and {args.y_out}")
np.save(f"{args.x_out}.npy", x_out)
np.save(f"{args.y_out}.npy", y_out)
print("[INFO] Saved.")
